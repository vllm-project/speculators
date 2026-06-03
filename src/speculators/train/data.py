import json
import math
import os
import random
import shutil
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Literal, cast

import openai
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_from_disk
from safetensors.torch import load_file
from torch.utils.data import Dataset
from transformers import AutoConfig

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
    wait_for_lock,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]
MULTIMODAL_SIDECAR_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grids",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
)
NON_TRAINING_KEYS = {
    *MULTIMODAL_SIDECAR_KEYS,
    "messages",
    "messages_json",
    "mm_file",
    "use_audio_in_video",
}


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            if not file.endswith("pt"):
                continue
            file_path = Path(root) / file
            datapath.append(file_path)

    return datapath


def slice_and_pad_to_length(tensor, length):
    sliced_tensor = tensor[:length]
    padding = [0, 0] * sliced_tensor.dim()
    padding[-1] = length - sliced_tensor.shape[0]
    return F.pad(sliced_tensor, padding)


def pad_last_dim_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
    sliced_tensor = tensor[..., :length]
    pad_amount = length - sliced_tensor.shape[-1]
    if pad_amount <= 0:
        return sliced_tensor
    padding = [0, pad_amount]
    padding.extend([0, 0] * (sliced_tensor.dim() - 1))
    return F.pad(sliced_tensor, padding)


def split_files(datapath: str, ratio: float = 0.9, seed: int = 0):
    """Given a datapath, split the files into a training and validation set
    ratio is the proportion of files to put in the training set
    1 - ratio is the proportion of files to put in the validation set
    """
    random.seed(seed)
    file_list = list_files(datapath)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train_files = int(num_files * ratio)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]
    return train_files, val_files


# Data standardization functions
StandardizeFnSig = Callable[[dict[str, Any]], dict[str, Any]]


def create_empty_sample(hidden_size: int, dtype: torch.dtype = torch.bfloat16):
    # data structure: {
    #     "hidden_states": [seq_len, 3 * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, 3 * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def standardize_data_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_states": [
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    ...
    #  ],
    # }

    return {
        "hidden_states": torch.cat(data["hidden_states"][:-1], dim=-1),
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["hidden_states"][-1],
        "loss_mask": data["loss_mask"],
    }


def _collect_mm_payload_from_messages(
    messages: list[dict],
) -> dict[str, list[str]]:
    """Flatten image/video/audio URLs out of vLLM-style chat messages.

    Why this exists (the "off-by-one newline" problem):

    ``prepare_data.py`` tokenizes multimodal conversations via the HF
    processor's ``apply_chat_template(tokenize=True)`` and persists the
    resulting ``input_ids`` into the Arrow dataset.  If we then request
    hidden states via ``chat.completions.create(messages=...)``, vLLM
    **re-renders** the same chat template server-side.  Multimodal chat
    templates (e.g. Qwen3-VL / Qwen3.6) insert media placeholder tokens
    and surrounding whitespace slightly differently across library versions,
    which causes an off-by-one token mismatch (typically a stray ``\\n``
    at a vision-placeholder boundary).  The strict equality check in
    ``extract_output`` then rejects every sample → loss=0 for the epoch.

    The fix is to bypass server-side template rendering entirely: send the
    pre-tokenized ``input_ids`` via the Completions API and attach the raw
    media URLs through ``extra_body.multi_modal_data`` so vLLM can still
    run the vision encoder without re-tokenizing.

    This function extracts those media URLs from the persisted ``messages``
    (which ``prepare_data.py`` already formatted as vLLM-compatible
    ``{type: "image_url", image_url: {url: "file:///..."}}`` dicts).
    """
    bucket: dict[str, list[str]] = {}
    for turn in messages:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for seg in content:
            if not isinstance(seg, dict):
                continue
            seg_type = seg.get("type", "")
            for modality in ("image", "video", "audio"):
                if seg_type == f"{modality}_url":
                    url = (seg.get(f"{modality}_url") or {}).get("url")
                    if url:
                        bucket.setdefault(modality, []).append(url)
    return bucket


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a vLLM ClientItem from a preprocessed dataset row.

    The default path is **token-id Completions**: we send ``input_ids``
    directly so the server does not re-render the chat template, which
    avoids the off-by-one token drift between ``prepare_data.py``'s HF
    processor tokenization and vLLM's server-side chat template rendering
    (see the docstring on ``_collect_mm_payload_from_messages`` for the full
    root-cause analysis).

    For multimodal samples, media URLs are extracted from the persisted
    ``messages_json`` and forwarded via ``multi_modal_data`` in the
    Completions request ``extra_body``.  This lets vLLM attach vision/audio
    encoder features to the exact token positions without re-tokenizing.

    ``messages`` is still populated as a fallback for code paths that
    explicitly opt into ``use_chat_completions=True`` (not recommended for
    multimodal — see above).
    """
    out_dict: dict[str, Any] = {}
    input_ids = _maybe_tensor(dataset_item["input_ids"], torch.long)
    out_dict["input_ids"] = input_ids.tolist() if input_ids is not None else []

    messages_json = dataset_item.get("messages_json", "")
    messages: list[dict] | None = None
    if messages_json:
        messages = json.loads(messages_json)
    elif "messages" in dataset_item:
        messages = dataset_item["messages"]

    if messages:
        # Keep messages around as a fallback path
        out_dict["messages"] = messages
        # Primary path: extract MM URLs so we can use Completions API
        mm = _collect_mm_payload_from_messages(messages)
        if mm:
            out_dict["multi_modal_data"] = mm

    return cast("ClientItem", out_dict)


def _maybe_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _batchify_mm_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 0:
        return value.view(1)
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _has_multimodal_payload(data: dict[str, Any]) -> bool:
    return any(data.get(key) is not None for key in MULTIMODAL_SIDECAR_KEYS)


def _make_rope_index_fn(verifier_name_or_path: str):
    """Build a callable producing 3D MRoPE position ids for Qwen multimodal models."""
    try:
        verifier_root_config = AutoConfig.from_pretrained(
            verifier_name_or_path, trust_remote_code=True
        )
    except Exception:  # noqa: BLE001
        return None

    thinker_config = getattr(verifier_root_config, "thinker_config", None)
    if thinker_config is not None:
        return _make_rope_index_fn_qwen3_omni(thinker_config)

    top_vision_config = getattr(verifier_root_config, "vision_config", None)
    top_text_config = getattr(verifier_root_config, "text_config", None)
    top_image_token_id = getattr(verifier_root_config, "image_token_id", None)
    if (
        top_vision_config is not None
        and top_text_config is not None
        and top_image_token_id is not None
    ):
        return _make_rope_index_fn_qwen3_5_moe(verifier_root_config)

    return None


def _make_rope_index_fn_qwen3_omni(thinker_config):
    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: PLC0415
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    except ImportError:
        return None

    vision_config = getattr(thinker_config, "vision_config", None)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 1)
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=getattr(thinker_config, "image_token_id", None),
            video_token_id=getattr(thinker_config, "video_token_id", None),
            audio_token_id=getattr(thinker_config, "audio_token_id", None),
            vision_start_token_id=getattr(
                thinker_config, "vision_start_token_id", None
            ),
            audio_start_token_id=getattr(thinker_config, "audio_start_token_id", None),
            position_id_per_seconds=getattr(
                thinker_config, "position_id_per_seconds", 25
            ),
        ),
        spatial_merge_size=spatial_merge_size,
    )
    dummy.get_llm_pos_ids_for_vision = MethodType(
        Qwen3OmniMoeThinkerForConditionalGeneration.get_llm_pos_ids_for_vision,
        dummy,
    )
    return MethodType(
        Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index,
        dummy,
    )


def _make_rope_index_fn_qwen3_5_moe(root_config):
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
            Qwen3_5MoeModel,
        )
    except ImportError:
        return None

    image_token_id = getattr(root_config, "image_token_id", None)
    video_token_id = getattr(root_config, "video_token_id", None)
    vision_config = getattr(root_config, "vision_config", None)
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        ),
    )
    dummy.get_vision_position_ids = MethodType(
        Qwen3_5MoeModel.get_vision_position_ids, dummy
    )
    raw_get_rope_index = MethodType(Qwen3_5MoeModel.get_rope_index, dummy)

    def adapter(
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **_ignored,
    ):
        ids = input_ids
        image_mask = (
            ids == image_token_id
            if image_token_id is not None
            else torch.zeros_like(ids, dtype=torch.bool)
        )
        video_mask = (
            ids == video_token_id
            if video_token_id is not None
            else torch.zeros_like(ids, dtype=torch.bool)
        )
        mm_token_type_ids = torch.zeros_like(ids, dtype=torch.int32)
        mm_token_type_ids[image_mask] = 1
        mm_token_type_ids[video_mask] = 2
        return raw_get_rope_index(
            input_ids=ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    return adapter


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index):
        raise NotImplementedError

    def _build_position_ids(self, data: dict[str, Any], seq_len: int) -> torch.Tensor:
        rope_index_fn = getattr(self, "_rope_index_fn", None)
        if rope_index_fn is not None and _has_multimodal_payload(data):
            audio_seqlens = _maybe_tensor(data.get("audio_feature_lengths"), torch.long)
            if audio_seqlens is None and data.get("feature_attention_mask") is not None:
                audio_seqlens = _maybe_tensor(
                    data["feature_attention_mask"], torch.long
                ).sum(dim=-1)

            position_ids, _ = rope_index_fn(
                input_ids=data["input_ids"].unsqueeze(0),
                image_grid_thw=_batchify_mm_tensor(
                    _maybe_tensor(data.get("image_grid_thw"), torch.long)
                ),
                video_grid_thw=_batchify_mm_tensor(
                    _maybe_tensor(data.get("video_grid_thw"), torch.long)
                ),
                use_audio_in_video=bool(data.get("use_audio_in_video", False)),
                audio_seqlens=audio_seqlens,
                second_per_grids=_maybe_tensor(data.get("second_per_grids")),
                attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            )
            return position_ids[:, 0].to(dtype=torch.long)

        return torch.arange(seq_len, dtype=torch.long)

    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)

        if data is None:
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Convert hidden states to the correct dtype
        data = {
            k: v.to(self.hidden_states_dtype) if "hidden_states" in k else v
            for k, v in data.items()
        }

        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = self._build_position_ids(data, seq_len)
        # shape: [seq_len] or [3, seq_len] for MRoPE multimodal samples

        # Keep multimodal payload only for rope index / HS generation
        for key in NON_TRAINING_KEYS:
            data.pop(key, None)

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        # Apply transform
        if self.transform:
            data = self.transform(data)

        return data


def _maybe_load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        hidden_states_path: str | PathLike | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        verifier_name_or_path: str | None = None,
    ):
        """Initialize the ArrowDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory that contains the preprocessed
            arrow dataset.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
        """
        self.datapath = Path(datapath)
        # ``prepare_data.py`` persists ``set_format(type="torch",
        # columns=["input_ids","loss_mask","seq_len"])`` into state.json via
        # save_to_disk.  That silently hides multimodal metadata columns
        # (``mm_file``, ``messages_json``, ``use_audio_in_video``) on every
        # subsequent ``load_from_disk`` call.  ``.with_format(None)`` is an
        # O(1) metadata reset (no tensor copy) that makes all stored columns
        # accessible again.
        self.data = load_from_disk(datapath).with_format(None)
        self.start_file_idx = 0
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            split_idx = int(len(self.data) * split_ratio)
            self.data = self.data.select(range(split_idx))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.start_file_idx = split_idx
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0.")

        self.hidden_states_path: Path = (
            self.datapath / "hidden_states"
            if hidden_states_path is None
            else Path(hidden_states_path)
        )
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.verifier_name_or_path = verifier_name_or_path or model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._rope_index_fn = (
            _make_rope_index_fn(self.verifier_name_or_path)
            if self.verifier_name_or_path is not None
            else None
        )

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def _setup_client(self):
        # Delay client setup so it runs in dataloader thread if on_missing="generate"
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = self.client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _maybe_generate_hs(
        self, index: int, *, is_multimodal: bool = False
    ) -> dict[str, torch.Tensor] | None:
        """Generate hidden states on-demand via the vLLM endpoint.

        For multimodal samples (``is_multimodal=True``), uses Chat Completions
        so vLLM runs its vision encoder.  The server's ``prompt_token_ids``
        become the authoritative ``input_ids`` (stored under the key
        ``"token_ids"`` in the returned dict) because the server may tokenize
        multimodal chat templates slightly differently from the HF processor
        used at prepare_data time.
        """
        if not self.client:
            self._setup_client()

        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)

        # Multimodal samples must go through Chat Completions so the vLLM
        # server runs the vision encoder.  The trade-off: vLLM re-renders
        # the chat template, so token_ids may differ from prepare_data's
        # output.  We accept vLLM's token_ids as truth (they are
        # positionally aligned with the hidden states it produces).
        use_chat = is_multimodal and client_item.get("messages") is not None

        try:
            result = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
                use_chat_completions=use_chat,
            )

            if use_chat:
                hs_filepath, server_token_ids = result  # type:ignore[misc]
            else:
                hs_filepath = result  # type:ignore[assignment]
                server_token_ids = None

            loaded_hs = _maybe_load_hs_file(Path(hs_filepath))

            # Overwrite token_ids with the server's authoritative version
            # so downstream _get_raw_data uses the correct positionally-
            # aligned sequence for loss/hidden-state pairing.
            if loaded_hs is not None and server_token_ids is not None:
                loaded_hs["token_ids"] = torch.tensor(
                    server_token_ids, dtype=torch.long
                )

            match self.on_generate:
                case "cache":
                    file_idx = self._map_to_file_idx(index)
                    target_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
                    # ``shutil.move`` requires ``target_path.parent`` to exist.
                    # When users point ``--hidden-states-path`` at a fresh
                    # directory that prepare_data.py has not touched, the
                    # parent dir is missing and every sample fails with
                    # ``[Errno 2] No such file or directory`` — the
                    # exception is swallowed by the ``except`` below, every
                    # row returns None, collate substitutes
                    # ``create_empty_sample``, and the trainer silently
                    # logs ``loss=0`` for the entire epoch. Materialising
                    # the directory once per missing-sample call is cheap
                    # and idempotent.
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(hs_filepath, target_path)
                case "delete":
                    Path(hs_filepath).unlink()
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Failed to load/cache hidden states for sample {index}: {e}",
                stacklevel=1,
            )
            return None

        return loaded_hs

    def _get_raw_data(self, index):
        # Fetch the full row upfront so we can access both training tensors
        # (input_ids, loss_mask) and multimodal metadata (mm_file,
        # messages_json, use_audio_in_video) from the same object.
        # Because with_format(None) is active, columns come back as raw
        # Python/numpy types; _maybe_tensor normalises them to torch.Tensor.
        row = self.data[index]
        stored_input_ids = _maybe_tensor(row["input_ids"], torch.long)
        stored_loss_mask = _maybe_tensor(row["loss_mask"], torch.long)
        # Fast-fail: skip corrupted/incomplete rows before expensive HS I/O.
        if stored_input_ids is None or stored_loss_mask is None:
            return None

        is_multimodal = bool(row.get("messages_json", ""))

        file_idx = self._map_to_file_idx(index)
        hs_file = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        loaded_hs = _maybe_load_hs_file(hs_file)

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(
                        index, is_multimodal=is_multimodal
                    )
                case "skip":
                    return None
                case "warn":
                    warnings.warn(
                        f"Failed to load hidden states for sample {index}. Skipping...",
                        stacklevel=1,
                    )
                    return None
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {index}."
                    )

        if loaded_hs is None:
            return loaded_hs

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, 4, hidden_size]
        #   "token_ids": [seq_len]
        # }

        # For multimodal samples generated via Chat Completions, vLLM's
        # server-side tokenization is authoritative (it ran the vision
        # encoder on that exact token sequence). The stored input_ids from
        # prepare_data.py may differ by ±1 token at vision-placeholder
        # boundaries, so we trust loaded_hs["token_ids"] unconditionally.
        # For text-only samples (or pre-generated offline HS), we still
        # enforce strict equality to catch data corruption early.
        authoritative_input_ids = loaded_hs["token_ids"]
        if not is_multimodal and not torch.equal(authoritative_input_ids, stored_input_ids):
            warnings.warn(
                f"Token IDs mismatch for text sample {index}: "
                f"hs has {len(authoritative_input_ids)} tokens, "
                f"dataset has {len(stored_input_ids)} tokens. Skipping.",
                stacklevel=1,
            )
            return None

        # Use authoritative_input_ids (from HS file) as ground truth.
        # For multimodal, this is vLLM's tokenization (aligned with HS);
        # for text-only, it equals stored_input_ids (verified above).
        # loss_mask length must match; truncate/pad if server tokenized
        # slightly differently (common for multimodal ±1 token drift).
        seq_len_hs = len(authoritative_input_ids)
        if len(stored_loss_mask) != seq_len_hs:
            if len(stored_loss_mask) > seq_len_hs:
                stored_loss_mask = stored_loss_mask[:seq_len_hs]
            else:
                stored_loss_mask = torch.cat([
                    stored_loss_mask,
                    torch.zeros(seq_len_hs - len(stored_loss_mask), dtype=torch.long),
                ])

        data = {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": authoritative_input_ids,  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": stored_loss_mask,  # [seq_len]
            "messages_json": row.get("messages_json", ""),
            "mm_file": row.get("mm_file", ""),
            "use_audio_in_video": bool(row.get("use_audio_in_video", 0)),
        }

        mm_path = data["mm_file"]
        if mm_path:
            mm = load_file(self.datapath / mm_path)
            for key in MULTIMODAL_SIDECAR_KEYS:
                if key in mm:
                    data[key] = mm[key]

        return data


class SampleFileDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the SampleFileDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory. All `.pt` files in this directory
            or its subdirectories will be loaded and used as training data. MUTUALLY
            EXCLUSIVE with `file_list`.
            file_list: The list of explict file paths to load data from. These files
            must be in the format produced by the Speculators generation scripts.
            MUTUALLY EXCLUSIVE with `datapath`.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.

            Note: datapath or file_list must be provided, but not both.

        """

        if datapath is not None and file_list is not None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        if datapath is not None:
            file_list = list_files(datapath)

        if file_list is None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        self.data: list[str] = file_list

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples.

        First tries to load exact lengths from sample_lengths.json if available.
        Falls back to approximation based on file sizes.
        """
        # Look for the sample_lengths.json file
        sample_lengths_path = Path(self.data[0]).parent / "sample_lengths.json"
        if sample_lengths_path.exists():
            try:
                with sample_lengths_path.open() as f:
                    sample_lengths = json.load(f)
                # Extract file index from filename (e.g., data_42.pt -> 42)
                lengths = []
                for fname in self.data:
                    file_stem = Path(fname).stem
                    file_idx = file_stem.split("_")[-1]
                    lengths.append(sample_lengths[file_idx])
                return lengths
            except (KeyError, ValueError):
                pass

        # Fallback: approximate lengths from file sizes
        item_0 = self.__getitem__(0)
        if item_0 is None:
            raise ValueError(
                "Failed to load first element of datasets for length approximation"
            )
        lengths_0 = item_0["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def _get_raw_data(self, index):
        return standardize_data_v1(
            torch.load(
                self.data[index], mmap=True, weights_only=True, map_location="cpu"
            )
        )


def create_collate_fn(
    max_len: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            batch = [create_empty_sample(hidden_size, dtype=dtype)]

        collated_data = {}
        has_mrope = any(
            b["position_ids"].ndim == 2 for b in batch  # type: ignore[index]
        )
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "position_ids":
                continue

            # Concatenate the tensors along the seq (0th) dimension
            collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
            # shape: [total_seq_len, ...]

            if key != "lengths":
                # Slice and pad on seq (0th) dimension to max_len
                collated_data[key] = slice_and_pad_to_length(
                    collated_data[key], max_len
                ).unsqueeze(0)
                # shape: [1, max_len, ...]

        if has_mrope:
            # MRoPE samples carry position_ids of shape [3, seq_len] (T/H/W
            # channels). After shift_batch they remain [3, seq_len-1].
            # We concatenate along the seq dimension, pad to max_len, then
            # add a batch dim. Qwen3-Omni Thinker rotary (used by EAGLE3 when
            # the verifier carries rope_parameters.mrope_section) expects
            # ``position_ids`` shaped ``[3, batch, seq_len]`` — channels
            # first — NOT the HF Llama4 / Gemma3 ``[batch, 3, seq_len]``
            # convention. Picking the wrong layout silently broadcasts the
            # 3 channels through the attention reshape and blows up
            # ``o_proj`` with a feature dim of ``3 * num_heads * head_dim``
            # (e.g. 12288 instead of 4096 for Qwen3.6 draft heads).
            position_ids = []
            for sample in batch:  # type: ignore[assignment]
                pos = sample["position_ids"]
                if pos.ndim == 1:
                    # Text-only sample in a mixed batch: broadcast to 3 channels
                    pos = pos.unsqueeze(0).expand(3, -1)
                position_ids.append(pos)
            # shape of each: [3, sample_seq_len]; cat along seq dim → [3, total_seq_len]
            collated_positions = torch.cat(position_ids, dim=-1)
            # Pad seq dim to max_len → [3, max_len], then add batch dim at
            # axis 1 → [3, 1, max_len] for Qwen-Omni rotary.
            collated_data["position_ids"] = pad_last_dim_to_length(
                collated_positions, max_len
            ).unsqueeze(1)
        else:
            # Standard 1D position_ids: [seq_len] per sample.
            # Cat along seq dim → [total_seq_len], pad → [max_len], batch → [1, max_len]
            collated_positions = torch.cat([b["position_ids"] for b in batch], dim=0)  # type: ignore[index]
            collated_data["position_ids"] = slice_and_pad_to_length(
                collated_positions, max_len
            ).unsqueeze(0)

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data["lengths"]
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        collated_data["lengths"] = torch.tensor(new_lengths, dtype=torch.long)
        return collated_data

    return collate_fn
