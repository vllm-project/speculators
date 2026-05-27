# ruff: noqa: ERA001
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
from typing import Any, Literal

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
    generate_hidden_states,
    generate_hidden_states_multimodal,
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


def create_empty_sample(hidden_size: int):
    # data structure: {
    #     "hidden_states": [seq_len, 3 * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }

    return {
        "hidden_states": torch.empty(0, 3 * hidden_size),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size),
        "loss_mask": torch.empty(0, dtype=torch.long),
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
    """Build a callable producing 3D MRoPE position ids for multimodal verifiers.

    Supports two config layouts:

    1. **Qwen3-Omni Thinker** (nested ``thinker_config``): the verifier root
       config exposes ``thinker_config.{vision_config, text_config,
       image_token_id, video_token_id, audio_*}``. We bind
       ``Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index`` to a
       dummy namespace matching the required ``self`` attributes.

    2. **Qwen3.5 / Qwen3.6 MoE** (flat layout): the verifier root config
       exposes ``{text_config, vision_config, image_token_id, video_token_id,
       vision_start_token_id}`` at the top level and has **no**
       ``thinker_config``. We bind ``Qwen3_5MoeModel.get_rope_index`` — whose
       signature differs (takes ``mm_token_type_ids`` instead of the
       audio-related kwargs) — and wrap it in an adapter so the caller can
       keep using the Qwen3-Omni-style kwargs.

    Returns ``None`` (and silently falls back to 1D arange in
    ``BaseDataset._build_position_ids``) when:
      * ``AutoConfig.from_pretrained`` fails (bad path, missing model, etc.)
      * The verifier config carries no recognized multimodal layout
        (text-only verifier)
      * Transformers does not expose the matching modeling class

    This keeps text-only training unaffected by multimodal-specific imports.
    """
    try:
        verifier_root_config = AutoConfig.from_pretrained(
            verifier_name_or_path, trust_remote_code=True
        )
    except Exception:  # noqa: BLE001
        return None

    # --- Dispatch on config layout ------------------------------------------
    thinker_config = getattr(verifier_root_config, "thinker_config", None)
    if thinker_config is not None:
        return _make_rope_index_fn_qwen3_omni(thinker_config)

    # Flat multimodal layout (Qwen3.5 / Qwen3.6 MoE): requires BOTH text_config
    # and vision_config at the top level, plus an image_token_id. We don't
    # accept a bare text_config because pure-text decoder-only Qwen configs
    # (e.g. Qwen3ForCausalLM) have text_config too but no MRoPE indexing.
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
    """Return a bound ``get_rope_index`` for Qwen3-Omni Thinker verifiers."""
    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
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
    """Return a Qwen3-Omni-shaped ``get_rope_index`` callable for Qwen3.5/3.6 MoE.

    ``Qwen3_5MoeModel.get_rope_index`` has signature
    ``(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw,
    attention_mask)`` — it takes a ``mm_token_type_ids`` tensor marking each
    token as text (0), image (1), or video (2). The rest of the pipeline
    (``_build_position_ids``) still emits Qwen3-Omni-style kwargs
    (``use_audio_in_video``, ``audio_seqlens``, ``second_per_grids``). We
    return a thin adapter that builds ``mm_token_type_ids`` from ``input_ids``
    using top-level ``{image_token_id, video_token_id}`` and ignores the
    unused audio kwargs, so callers never need to know which verifier family
    they have.
    """
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
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
        # Build mm_token_type_ids from input_ids. Shape must match input_ids.
        ids = input_ids
        if image_token_id is not None:
            image_mask = ids == image_token_id
        else:
            image_mask = torch.zeros_like(ids, dtype=torch.bool)
        if video_token_id is not None:
            video_mask = ids == video_token_id
        else:
            video_mask = torch.zeros_like(ids, dtype=torch.bool)
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
        hidden_states_dtype=torch.float,
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

        # Anchor mask: positions where all RoPE channels coincide. For 3D MRoPE
        # the T/H/W channels diverge on vision/audio placeholder tokens, so we
        # must exclude those from the anchor candidate set to keep
        # ``mask_position_ids`` (computed on T only) safe to broadcast to H/W.
        pos = data["position_ids"]
        if pos.ndim == 2:
            data["anchor_mask"] = (
                (pos[0] == pos[1]) & (pos[1] == pos[2])
            ).to(torch.long)
        else:
            data["anchor_mask"] = torch.ones(seq_len, dtype=torch.long)

        # Keep multimodal payload only for rope index / HS generation
        for key in NON_TRAINING_KEYS:
            data.pop(key, None)

        # Apply transform
        if self.transform:
            data = self.transform(data)

        return data


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
        hidden_states_dtype=torch.float,
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
        self.data = load_from_disk(datapath)
        # ``prepare_data.py`` persists ``set_format(type="torch",
        # columns=["input_ids","loss_mask","seq_len"])`` into state.json via
        # save_to_disk. That silently hides the multimodal metadata columns
        # (``mm_file``, ``messages_json``, ``use_audio_in_video``) on every
        # subsequent ``load_from_disk`` call, causing ``row.get("mm_file", "")``
        # below to return ``""`` and the whole sidecar + 3D MRoPE branch to be
        # skipped. Reset the format so every stored column stays accessible.
        # ``with_format(None)`` is an O(1) metadata flip — no tensors copied.
        self.data = self.data.with_format(None)
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
                f"An explicit model name was passed ({self.model}) which doesn't match "
                f"found model_id {model_id}. "
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _maybe_load_hs_file(self, index: int) -> dict[str, torch.Tensor] | None:
        file_idx = self._map_to_file_idx(index)
        candidate_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        if candidate_path.exists():
            return load_file(candidate_path)

        return None

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:
        if not self.client:
            self._setup_client()

        row = self.data[index]
        input_ids = _maybe_tensor(row["input_ids"], torch.long)
        if input_ids is None:
            return None

        messages_json = row.get("messages_json", "")
        try:
            if messages_json:
                hs_filepath = generate_hidden_states_multimodal(
                    self.client,  # type:ignore[arg-type]
                    self.model,  # type:ignore[arg-type]
                    json.loads(messages_json),
                    timeout=self.request_timeout,
                    max_retries=self.max_retries,
                )
            else:
                hs_filepath = generate_hidden_states(
                    self.client,  # type:ignore[arg-type]
                    self.model,  # type:ignore[arg-type]
                    input_ids.tolist(),
                    timeout=self.request_timeout,
                    max_retries=self.max_retries,
                )
        except Exception as e:  # noqa: BLE001
            warnings.warn(str(e), stacklevel=1)
            return None

        loaded_hs = load_file(hs_filepath)

        match self.on_generate:
            case "cache":
                file_idx = self._map_to_file_idx(index)
                target_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
                shutil.move(hs_filepath, target_path)
            case "delete":
                Path(hs_filepath).unlink()

        return loaded_hs

    def _get_raw_data(self, index):
        row = self.data[index]
        stored_input_ids = _maybe_tensor(row["input_ids"], torch.long)
        stored_loss_mask = _maybe_tensor(row["loss_mask"], torch.long)
        if stored_input_ids is None or stored_loss_mask is None:
            return None

        loaded_hs = self._maybe_load_hs_file(index)

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(index)
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

        if not torch.equal(loaded_hs["token_ids"], stored_input_ids):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {stored_input_ids}",
                stacklevel=1,
            )
            return None

        data = {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": stored_input_ids,  # [seq_len]
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
        hidden_states_dtype=None,
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
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found
            batch = [create_empty_sample(hidden_size)]

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
            position_ids = []
            for sample in batch:  # type: ignore[assignment]
                pos = sample["position_ids"]
                if pos.ndim == 1:
                    pos = pos.unsqueeze(0).expand(3, -1)
                position_ids.append(pos)
            collated_positions = torch.cat(position_ids, dim=-1)
            collated_data["position_ids"] = pad_last_dim_to_length(
                collated_positions, max_len
            ).unsqueeze(1)
        else:
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
