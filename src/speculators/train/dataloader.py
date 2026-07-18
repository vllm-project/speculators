from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from speculators.data_generation.windowed_artifacts import (
    StreamSampleIndex,
    canonical_position_id,
)
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    SampleFileDataset,
    create_collate_fn,
    split_files,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


class _WindowedDataset(Protocol):
    windowed_artifacts_enabled: bool

    def configure_windowed_stream(self, sampler: Any) -> str: ...

    def windowed_request_id(self, dataset_index: int) -> str: ...


class WindowedBatchSampler:
    """Annotate sampler indices with stable positions in the consumed order."""

    def __init__(
        self,
        sampler: MultipackDistributedBatchSamplerV2,
        *,
        stream_id: str,
        request_id_for_index: Callable[[int], str],
    ) -> None:
        self.sampler = sampler
        self.stream_id = stream_id
        self.request_id_for_index = request_id_for_index
        self.epoch = sampler.epoch
        self._epoch_counts: dict[int, int] = {}
        self._cached_generated_batches: tuple[int, list[list[StreamSampleIndex]]] = (
            -1,
            [],
        )

    def _raw_batches(self, epoch: int) -> list[Any]:
        return self.sampler._generate_batches(epoch)  # noqa: SLF001

    def _epoch_count(self, epoch: int) -> int:
        if epoch not in self._epoch_counts:
            self._epoch_counts[epoch] = sum(
                len(batch) for batch in self._raw_batches(epoch)
            )
        return self._epoch_counts[epoch]

    def _sequence_offset(self, epoch: int) -> int:
        return sum(self._epoch_count(previous) for previous in range(epoch))

    def _generate_batches(self, epoch: int) -> list[list[StreamSampleIndex]]:
        if self._cached_generated_batches[0] == epoch:
            return self._cached_generated_batches[1]
        offset = self._sequence_offset(epoch)
        ordinal = 0
        batches: list[list[StreamSampleIndex]] = []
        for batch_ordinal, raw_batch in enumerate(self._raw_batches(epoch)):
            batch: list[StreamSampleIndex] = []
            batch_start_sequence = offset + ordinal
            batch_end_sequence = batch_start_sequence + len(raw_batch)
            for raw_index in raw_batch:
                dataset_index = int(raw_index)
                batch.append(
                    StreamSampleIndex(
                        stream_id=self.stream_id,
                        sequence=offset + ordinal,
                        epoch=epoch,
                        ordinal=ordinal,
                        dataset_index=dataset_index,
                        batch_ordinal=batch_ordinal,
                        batch_start_sequence=batch_start_sequence,
                        batch_end_sequence=batch_end_sequence,
                        request_id=self.request_id_for_index(dataset_index),
                        position_id=canonical_position_id(
                            self.stream_id,
                            epoch=epoch,
                            ordinal=ordinal,
                            dataset_index=dataset_index,
                            batch_ordinal=batch_ordinal,
                            batch_start_sequence=batch_start_sequence,
                            batch_end_sequence=batch_end_sequence,
                        ),
                    )
                )
                ordinal += 1
            batches.append(batch)
        self._epoch_counts[epoch] = ordinal
        self._cached_generated_batches = (epoch, batches)
        return batches

    def full_epoch_samples(self, epoch: int) -> tuple[StreamSampleIndex, ...]:
        return tuple(
            sample for batch in self._generate_batches(epoch) for sample in batch
        )

    def __iter__(self):
        return iter(self._generate_batches(self.epoch))

    def __len__(self) -> int:
        return len(self._generate_batches(self.epoch))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch)


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
) -> DataLoader:
    batch_sampler: Any = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
    )
    if getattr(dataset, "windowed_artifacts_enabled", False):
        windowed_dataset: _WindowedDataset = dataset  # type: ignore[assignment]
        stream_id = windowed_dataset.configure_windowed_stream(batch_sampler)
        batch_sampler = WindowedBatchSampler(
            batch_sampler,
            stream_id=stream_id,
            request_id_for_index=windowed_dataset.windowed_request_id,
        )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=create_collate_fn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    legacy_data: bool,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    shared_artifacts_path: str | None = None,
    shared_artifacts_namespace: str | None = None,
    shared_artifacts_ttl_seconds: float | None = 3600.0,
    shared_artifacts_lock_timeout_seconds: float = 300.0,
    shared_artifacts_consumer_id: str | None = None,
    shared_artifacts_lookbehind: int = 2,
    shared_artifacts_lookahead: int = 40,
    shared_artifacts_max_prefetch_per_consumer: int = 8,
    shared_artifacts_capture_batch_size: int = 8,
    shared_artifacts_capture_batch_wait_seconds: float = 0.002,
    shared_artifacts_max_inflight: int = 32,
    shared_artifacts_consumer_timeout_seconds: float = 120.0,
    shared_artifacts_claim_timeout_seconds: float = 300.0,
    shared_artifacts_generation_attempts: int = 3,
    train_data_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Handles dataset construction (legacy vs Arrow) and dataloader wiring.
    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.
    """
    noise_transform = AddUniformNoise(std=noise_std)

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    if legacy_data:
        warnings.warn(
            "Using '--legacy-data' is deprecated and will be removed soon.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        train_files, val_files = split_files(data_path, ratio=train_data_ratio)
        train_dataset: BaseDataset = SampleFileDataset(
            file_list=train_files,
            max_len=total_seq_len,
            transform=noise_transform,
            hidden_states_dtype=hidden_states_dtype,
        )
        val_dataset: BaseDataset = SampleFileDataset(
            file_list=val_files,
            max_len=total_seq_len,
            hidden_states_dtype=hidden_states_dtype,
        )
    else:
        train_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            transfer=transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            transform=noise_transform,
            split_ratio=train_data_ratio,
            model=verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
            shared_artifacts_path=shared_artifacts_path,
            shared_artifacts_namespace=shared_artifacts_namespace,
            shared_artifacts_ttl_seconds=shared_artifacts_ttl_seconds,
            shared_artifacts_lock_timeout_seconds=(
                shared_artifacts_lock_timeout_seconds
            ),
            shared_artifacts_consumer_id=(
                f"{shared_artifacts_consumer_id}:train"
                if shared_artifacts_consumer_id is not None
                else None
            ),
            shared_artifacts_lookbehind=shared_artifacts_lookbehind,
            shared_artifacts_lookahead=shared_artifacts_lookahead,
            shared_artifacts_max_prefetch_per_consumer=(
                shared_artifacts_max_prefetch_per_consumer
            ),
            shared_artifacts_capture_batch_size=shared_artifacts_capture_batch_size,
            shared_artifacts_capture_batch_wait_seconds=(
                shared_artifacts_capture_batch_wait_seconds
            ),
            shared_artifacts_max_inflight=shared_artifacts_max_inflight,
            shared_artifacts_consumer_timeout_seconds=(
                shared_artifacts_consumer_timeout_seconds
            ),
            shared_artifacts_claim_timeout_seconds=(
                shared_artifacts_claim_timeout_seconds
            ),
            shared_artifacts_generation_attempts=(shared_artifacts_generation_attempts),
        )
        val_dataset = ArrowDataset(
            datapath=data_path,
            max_len=total_seq_len,
            transfer=transfer,
            vllm_endpoint=vllm_endpoint,
            on_missing=on_missing,
            on_generate=on_generate,
            split_ratio=train_data_ratio - 1.0,
            model=verifier_name_or_path,
            hidden_states_dtype=hidden_states_dtype,
            request_timeout=request_timeout,
            max_retries=max_retries,
            shared_artifacts_path=shared_artifacts_path,
            shared_artifacts_namespace=shared_artifacts_namespace,
            shared_artifacts_ttl_seconds=shared_artifacts_ttl_seconds,
            shared_artifacts_lock_timeout_seconds=(
                shared_artifacts_lock_timeout_seconds
            ),
            shared_artifacts_consumer_id=(
                f"{shared_artifacts_consumer_id}:val"
                if shared_artifacts_consumer_id is not None
                else None
            ),
            shared_artifacts_lookbehind=shared_artifacts_lookbehind,
            shared_artifacts_lookahead=shared_artifacts_lookahead,
            shared_artifacts_max_prefetch_per_consumer=(
                shared_artifacts_max_prefetch_per_consumer
            ),
            shared_artifacts_capture_batch_size=shared_artifacts_capture_batch_size,
            shared_artifacts_capture_batch_wait_seconds=(
                shared_artifacts_capture_batch_wait_seconds
            ),
            shared_artifacts_max_inflight=shared_artifacts_max_inflight,
            shared_artifacts_consumer_timeout_seconds=(
                shared_artifacts_consumer_timeout_seconds
            ),
            shared_artifacts_claim_timeout_seconds=(
                shared_artifacts_claim_timeout_seconds
            ),
            shared_artifacts_generation_attempts=(shared_artifacts_generation_attempts),
        )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
    )

    return train_loader, val_loader
