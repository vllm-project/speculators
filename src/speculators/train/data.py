import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from speculators.train.data_types import DatasetRecord
from speculators.train.hidden_states_source import (
    HiddenStatesSource,
)
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    RecoveryMetadata,
    SampleUnavailable,
)

BatchType = dict[str, Any]


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


class ArrowDataset(Dataset):
    """Load preprocessed Arrow rows and expose them as dataset records."""

    def __init__(
        self,
        datapath: str | PathLike,
        train_ratio: float = 1.0,
        split: Literal["train", "val"] = "train",
    ):
        self.datapath = Path(datapath)
        self.data = load_from_disk(self.datapath)
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
                f"train_ratio={train_ratio} gives split_idx={split_idx}"
            )
        self.start_storage_index = start
        self.data = self.data.select(range(start, stop))
        self.approx_lengths = list(self.data.with_format(None)["seq_len"])

    def _map_to_storage_index(self, index: int) -> int:
        return index + self.start_storage_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> DatasetRecord:
        return DatasetRecord.from_arrow_row(
            self.data[index],
            storage_index=self._map_to_storage_index(index),
        )


class SpeculatorsDataset(Dataset):
    """Combine Arrow records with hidden states for training."""

    def __init__(
        self,
        max_len: int,
        records: ArrowDataset,
        hidden_states: HiddenStatesSource,
        hidden_states_dtype=torch.bfloat16,
        num_target_layers: int = 3,
    ):
        self.max_len = max_len
        self.hidden_states_dtype = hidden_states_dtype
        self.records = records
        self.num_target_layers = num_target_layers
        self.hidden_states = hidden_states
        self.approx_lengths = records.approx_lengths

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BatchType | SampleUnavailable:
        record = self.records[index]
        loaded_hs = self.hidden_states.get_one(record)

        if isinstance(loaded_hs, SampleUnavailable):
            return loaded_hs

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        if not torch.equal(loaded_hs["token_ids"], record.input_ids):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {record.input_ids}",
                stacklevel=1,
            )
            return SampleUnavailable(
                reason=f"Extracted token ids do not match sample {index}"
            )

        # The payload carries one entry per captured layer, ordered by layer id,
        # and its trailing entry is the verifier's final hidden state -- the input
        # of `verifier_lm_head`, i.e. the distilling target.
        #
        # The capture list normally ends one layer past the draft's target layers
        # (launch_vllm.py appends the final layer unless it is already listed), so
        # the payload is one entry wider than `fc` and the trailing entry is
        # split off. When the capture list already ends at the verifier's final
        # layer, the launcher appends nothing and the payload is exactly as wide
        # as `fc`: every entry feeds `fc` and the trailing one doubles as the
        # verifier's final hidden state. Any other width keeps the original
        # behaviour of dropping the trailing entry, so no configuration that ran
        # before this change can start failing.
        num_payload_layers = loaded_hs["hidden_states"].shape[1]
        if num_payload_layers == self.num_target_layers:
            hs_features = loaded_hs["hidden_states"]
        else:
            hs_features = loaded_hs["hidden_states"][:, :-1]

        data = {
            "hidden_states": hs_features.flatten(1),  # [seq_len, n_feat * h]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": record.loss_mask,  # [seq_len]
        }

        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        return data

    def close(self) -> None:
        """Close resources owned by the hidden-state source."""
        self.hidden_states.close()


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
