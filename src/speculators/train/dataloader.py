from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    SampleFileDataset,
    CollateFn,
    split_files,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=CollateFn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
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
