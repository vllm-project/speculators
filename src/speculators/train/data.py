import logging
import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
)
from speculators.train.noise_transforms import TransformTensors
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    GenerationRecoveryGuard,
    RecoveryMetadata,
    SampleUnavailable,
)

BatchType = dict[str, Any]
logger = logging.getLogger("speculators")


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
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


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
    out_dict: dict = {"input_ids": dataset_item["input_ids"].tolist()}

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]

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

    def __getitem__(self, index) -> BatchType | SampleUnavailable:
        data = self._get_raw_data(index)

        if isinstance(data, SampleUnavailable):
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

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


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        train_ratio: float = 1.0,
        split: Literal["train", "val"] = "train",
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        generation_validation_retries: int = 2,
        max_consecutive_generation_failures: int = 20,
    ):
        self.data = load_from_disk(datapath)
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0.0, 1.0], got {train_ratio}")
        if split == "val" and train_ratio == 1.0:
            raise ValueError("train_ratio=1.0 leaves no validation split")

        # Both splits derive their boundary from this one expression,
        # so they are exactly complementary.
        split_idx = int(len(self.data) * train_ratio)
        start, stop = (
            (0, split_idx) if split == "train" else (split_idx, len(self.data))
        )
        if start >= stop:
            raise ValueError(
                f"{split} split is empty (dataset has {len(self.data)} rows, "
                f"train_ratio={train_ratio} gives split_idx={split_idx})"
            )
        self.start_file_idx = start
        self.data = self.data.select(range(start, stop))

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.generation_recovery = GenerationRecoveryGuard(
            retries=generation_validation_retries,
            max_consecutive_failures=max_consecutive_generation_failures,
        )

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def _setup_client(self):
        client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()
        # Do not retain a half-initialized client if model discovery or Mooncake
        # setup failed; the outer full-round-trip retry should redo initialization.
        self.client = client

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _cleanup_failed_handle(self, handle: str | None) -> None:
        if handle is None:
            return
        try:
            self.transfer.delete(handle)
        except Exception as cleanup_error:  # noqa: BLE001 - recovery must be best effort
            logger.warning(
                "Failed to clean generated hidden-state handle %s: %s",
                handle,
                cleanup_error,
            )

    def _generate_hs_round_trip(
        self,
        index: int,
        dataset_item: dict,
        client_item: ClientItem,
    ) -> dict[str, torch.Tensor]:
        handle: str | None = None
        try:
            if not self.client:
                self._setup_client()
            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(handle)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest/checksum validation first.
            check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())

            file_idx = self._map_to_file_idx(index)
            if self.on_generate == "cache":
                self.transfer.cache(handle, file_idx)
            else:
                try:
                    self.transfer.delete(handle)
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Loaded a valid hidden-state sample but failed to delete "
                        "handle %s: %s",
                        handle,
                        cleanup_error,
                    )
            return loaded_hs
        except Exception:
            self._cleanup_failed_handle(handle)
            raise

    def _maybe_generate_hs(
        self, index: int
    ) -> dict[str, torch.Tensor] | SampleUnavailable:
        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)
        file_idx = self._map_to_file_idx(index)
        return self.generation_recovery.run(
            lambda: self._generate_hs_round_trip(index, dataset_item, client_item),
            description=(
                f"Hidden-state round trip failed for dataset index {index}, "
                f"file index {file_idx}"
            ),
        )

    def _get_raw_data(self, index):
        file_idx = self._map_to_file_idx(index)
        loaded_hs: dict[str, torch.Tensor] | SampleUnavailable | None
        loaded_hs = self.transfer.get_cached(file_idx)

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(index)
                case "skip":
                    return SampleUnavailable()
                case "warn":
                    warnings.warn(
                        f"Failed to load hidden states for sample {index}. Skipping...",
                        stacklevel=1,
                    )
                    return SampleUnavailable(
                        reason=f"Hidden states unavailable for sample {index}"
                    )
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {index}."
                    )

        if isinstance(loaded_hs, SampleUnavailable):
            return loaded_hs
        if loaded_hs is None:
            return SampleUnavailable(
                reason=f"Hidden states unavailable for sample {index}"
            )

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        if not torch.equal(loaded_hs["token_ids"], self.data[index]["input_ids"]):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {self.data[index]['input_ids']}",
                stacklevel=1,
            )
            return SampleUnavailable(
                reason=f"Cached token ids do not match sample {index}"
            )

        return {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": self.data[index]["loss_mask"],  # [seq_len]
        }


class CollateFn:
    """Picklable collate function for use with ``multiprocessing_context='spawn'``."""

    def __init__(
        self,
        max_len: int,
        hidden_size: int,
        num_target_layers: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        preprocess: Callable[[BatchType], BatchType] | None = None,
    ):
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_target_layers = num_target_layers
        self.dtype = dtype
        self.preprocess = preprocess

    def _clean_batch(
        self, batch: Sequence[BatchType | SampleUnavailable | None]
    ) -> tuple[list[BatchType], list[SampleUnavailable], int]:
        """Preprocess valid samples and collect unavailable and dropped samples."""
        preprocess = self.preprocess
        unavailable = []
        num_dropped = 0
        new_batch = []
        for item in batch:
            if item is None:
                num_dropped += 1
                continue
            if isinstance(item, SampleUnavailable):
                unavailable.append(item)
                num_dropped += 1
                continue

            new_batch.append(preprocess(item) if preprocess else item)

        return new_batch, unavailable, num_dropped

    def __call__(
        self, batch: Sequence[BatchType | SampleUnavailable | None]
    ) -> BatchType:
        max_len = self.max_len
        dtype = self.dtype

        batch, unavailable, num_dropped = self._clean_batch(batch)

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(
                self.hidden_size, self.num_target_layers, dtype=dtype
            )
            if self.preprocess:
                empty = self.preprocess(empty)
            batch = [empty]
            locally_empty = True
        else:
            locally_empty = False

        collated_data: BatchType = {}
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "lengths":
                collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
                continue
            # one copy per sample: preallocated buffer, hidden states cast during write
            first = batch[0][key]  # type: ignore[index]
            buffer_dtype = dtype if "hidden_states" in key else first.dtype
            out = torch.zeros(
                (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
            )
            offset = 0
            for b in batch:
                tensor = b[key]  # type: ignore[index]
                num_rows = min(tensor.shape[0], max_len - offset)
                out[offset : offset + num_rows] = tensor[:num_rows]
                offset += num_rows
                if offset == max_len:
                    break
            collated_data[key] = out.unsqueeze(0)
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

        collated_data["error_records"] = num_dropped
        metadata = RecoveryMetadata.from_unavailable(
            unavailable,
            locally_empty=locally_empty,
        )
        if metadata.failure_count or metadata.locally_empty:
            collated_data[RECOVERY_METADATA_KEY] = metadata

        return collated_data
