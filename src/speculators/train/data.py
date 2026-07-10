import json
import math
import os
import random
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_from_disk
from safetensors.torch import load_file
from torch.utils.data import Dataset

from speculators.data_generation.offline import (
    align_hidden_states_to_tokens,
    atomic_move_safetensors,
    atomic_save_safetensors,
    check_hidden_states,
    durable_unlink_safetensors,
    hidden_states_file_sha256,
    validate_generated_source_ownership,
    validate_hidden_states_path,
    validate_hidden_states_root,
    validate_hidden_states_tensors,
)
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
    wait_for_lock,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]


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


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used for explicitly skipped samples, the implicit
    # float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
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


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    multimodal_types = {
        "audio",
        "audio_url",
        "image",
        "image_url",
        "input_audio",
        "input_image",
        "input_video",
        "video",
        "video_url",
    }
    text_types = {None, "input_text", "text"}

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                return True
            part_type = part.get("type")
            if part_type in multimodal_types:
                return True
            if part_type not in text_types:
                return True

    return False


def _parse_client_tools(raw_tools: object) -> list[dict[str, Any]] | None:
    """Parse the canonical tools column for a Chat Completions request."""
    if raw_tools is None or raw_tools == "":
        return None

    parsed_tools: object
    if isinstance(raw_tools, str):
        try:
            parsed_tools = json.loads(raw_tools)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON in preprocessed tools column") from e
    elif isinstance(raw_tools, list):
        # Backward compatibility for in-memory/older datasets that stored the
        # parsed tool list instead of canonical JSON.
        parsed_tools = raw_tools
    else:
        raise ValueError(
            "Invalid preprocessed tools column: expected canonical JSON or a list"
        )

    if not isinstance(parsed_tools, list) or not all(
        isinstance(tool, dict) for tool in parsed_tools
    ):
        raise ValueError(
            "Invalid preprocessed tools schema: expected a list of objects"
        )

    return cast("list[dict[str, Any]]", parsed_tools) or None


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    input_ids = dataset_item["input_ids"]
    out_dict: dict = {
        "input_ids": input_ids.tolist()
        if hasattr(input_ids, "tolist")
        else list(input_ids)
    }

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]
        tools = _parse_client_tools(dataset_item.get("tools"))
        if tools is not None:
            out_dict["tools"] = tools

    return cast("ClientItem", out_dict)


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

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

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


def _maybe_load_hs_file(
    file_path: Path,
    allowed_root: Path,
    *,
    allow_missing_root: bool = False,
) -> dict[str, torch.Tensor] | None:
    if allow_missing_root and not allowed_root.exists():
        validate_hidden_states_root(allowed_root, require_exists=False)
        return None

    file_path = validate_hidden_states_path(
        file_path,
        allowed_root,
        require_exists=False,
    )
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        if Path(lock_path).is_symlink():
            raise ValueError(f"Hidden-state lock path is a symlink: {lock_path}")
        wait_for_lock(lock_path)

    if file_path.exists():
        file_path = validate_hidden_states_path(file_path, allowed_root)
        return load_file(file_path)

    return None


def _load_generated_hs_file_with_stable_digest(
    file_path: Path,
    allowed_root: Path,
) -> tuple[dict[str, torch.Tensor], str]:
    """Load a generated source and prove its bytes stayed stable while read."""
    file_path = validate_hidden_states_path(
        file_path,
        allowed_root,
        require_exists=False,
    )
    lock_path = Path(str(file_path) + ".lock")
    if lock_path.exists():
        if lock_path.is_symlink():
            raise ValueError(f"Hidden-state lock path is a symlink: {lock_path}")
        wait_for_lock(str(lock_path))

    file_path = validate_hidden_states_path(file_path, allowed_root)
    before = hidden_states_file_sha256(file_path, allowed_root=allowed_root)
    loaded = load_file(file_path)
    after = hidden_states_file_sha256(file_path, allowed_root=allowed_root)
    if before != after:
        raise RuntimeError(
            "Generated hidden-state source changed while it was being loaded"
        )
    return loaded, before


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
    ):
        """Initialize the ArrowDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory that contains the preprocessed
            arrow dataset.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
        """
        self.data = load_from_disk(datapath)
        self.start_file_idx = 0
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.start_file_idx = 0
            split_idx = int(len(self.data) * split_ratio)
            self.data = self.data.select(range(split_idx))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.start_file_idx = split_idx
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0.")

        self.hidden_states_path: Path = (
            Path(datapath) / "hidden_states"
            if hidden_states_path is None
            else Path(hidden_states_path)
        )
        self.hidden_states_path = validate_hidden_states_root(
            self.hidden_states_path,
            require_exists=False,
        )
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        if self.on_generate == "cache" or self.on_missing == "generate":
            self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        if self.hidden_states_path.exists():
            self.hidden_states_path = validate_hidden_states_root(
                self.hidden_states_path
            )
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries

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

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor]:
        if not self.client:
            self._setup_client()

        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)

        hs_filepath = generate_hidden_states(
            self.client,  # type:ignore[arg-type]
            self.model,  # type:ignore[arg-type]
            client_item,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

        source_path = validate_hidden_states_path(
            hs_filepath,
            self.hidden_states_path,
            require_exists=False,
        )
        loaded_hs, source_sha256 = _load_generated_hs_file_with_stable_digest(
            source_path,
            self.hidden_states_path,
        )

        file_idx = self._map_to_file_idx(index)
        target_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        source_path = validate_hidden_states_path(
            source_path,
            self.hidden_states_path,
        )
        validate_generated_source_ownership(
            source_path,
            target_path,
            source_root=self.hidden_states_path,
            target_root=self.hidden_states_path,
            allow_current_target=(
                self.on_generate == "cache" and "messages" in client_item
            ),
        )

        loaded_hs, truncated = align_hidden_states_to_tokens(
            loaded_hs,
            client_item["input_ids"],
            allow_prefix_truncation="messages" in client_item,
        )
        validate_generated_source_ownership(
            source_path,
            target_path,
            source_root=self.hidden_states_path,
            target_root=self.hidden_states_path,
            allow_current_target=self.on_generate == "cache" and truncated,
        )

        match self.on_generate:
            case "cache":
                if truncated:
                    source_path = validate_hidden_states_path(
                        source_path, self.hidden_states_path
                    )
                    atomic_save_safetensors(
                        {key: value.contiguous() for key, value in loaded_hs.items()},
                        target_path,
                        allowed_root=self.hidden_states_path,
                        allow_replace=source_path == target_path,
                        expected_existing_sha256=(
                            source_sha256 if source_path == target_path else None
                        ),
                    )
                    if source_path != target_path:
                        durable_unlink_safetensors(
                            source_path,
                            allowed_root=self.hidden_states_path,
                            expected_sha256=source_sha256,
                        )
                elif source_path != target_path:
                    atomic_move_safetensors(
                        source_path,
                        target_path,
                        source_root=self.hidden_states_path,
                        target_root=self.hidden_states_path,
                        expected_source_sha256=source_sha256,
                        expected_tokens=client_item["input_ids"],
                    )
            case "delete":
                durable_unlink_safetensors(
                    source_path,
                    allowed_root=self.hidden_states_path,
                    expected_sha256=source_sha256,
                )

        return loaded_hs

    def _get_raw_data(self, index):
        file_idx = self._map_to_file_idx(index)
        candidate_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        loaded_hs = _maybe_load_hs_file(
            candidate_path,
            self.hidden_states_path,
            allow_missing_root=True,
        )

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

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        dataset_item = self.data[index]
        expected_tokens = [int(token) for token in dataset_item["input_ids"]]
        try:
            validate_hidden_states_tensors(loaded_hs)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid hidden-state cache for sample {index}: {error}"
            ) from error

        actual_tokens = [int(token) for token in loaded_hs["token_ids"].tolist()]
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f" match input ids {expected_tokens}"
            )
        try:
            check_hidden_states(
                loaded_hs,
                expected_tokens,
            )
        except ValueError as error:
            raise RuntimeError(
                f"Invalid hidden-state cache for sample {index}: {error}"
            ) from error

        return {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"].to(torch.long),  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": torch.as_tensor(
                dataset_item["loss_mask"], dtype=torch.long
            ),  # [seq_len]
        }


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
    num_target_layers: int = 3,
    dtype: torch.dtype | None = None,
    hidden_states_dtype: torch.dtype | None = None,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    if (
        dtype is not None
        and hidden_states_dtype is not None
        and dtype != hidden_states_dtype
    ):
        raise ValueError("dtype and hidden_states_dtype must match when both are set")
    dtype = dtype or hidden_states_dtype or torch.bfloat16

    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(hidden_size, num_target_layers, dtype=dtype)
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        collated_data = {}
        for key in batch[0]:  # type: ignore[union-attr]
            # Concatenate the tensors along the seq (0th) dimension
            collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
            # shape: [total_seq_len, ...]

            if key != "lengths":
                # Slice and pad on seq (0th) dimension to max_len
                collated_data[key] = slice_and_pad_to_length(
                    collated_data[key], max_len
                ).unsqueeze(0)
                # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids

        return collated_data

    return collate_fn
