from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    CollateFn,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    torch.set_num_threads(1)


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    no_packing: bool = False,
) -> DataLoader:
    lengths = dataset.approx_lengths
    if no_packing:
        # Feed the packer a constant length equal to the whole token budget, so
        # every batch holds exactly one conversation per rank. That makes the
        # global batch a count of CONVERSATIONS (world_size x accumulation_steps)
        # instead of a token budget, which is the unit a recipe is specified in.
        lengths = np.full(len(lengths), int(total_seq_len), dtype=np.int64)
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=lengths,
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
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
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
    no_packing: bool = False,
    val_data_path: str | None = None,
    val_transfer: HiddenStatesTransfer | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.
    """
    _limit_worker_threads()
    noise_transform = AddUniformNoise(std=noise_std)

    # With a dedicated validation corpus, training uses ALL of data_path and
    # evaluates on the separate file, so there is no ratio split to make.
    use_val_path = val_data_path is not None
    if use_val_path:
        train_data_ratio = 1.0
    elif not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    train_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        transform=noise_transform,
        train_ratio=train_data_ratio,
        split="train",
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )
    val_dataset: BaseDataset = ArrowDataset(
        datapath=data_path if val_data_path is None else val_data_path,
        max_len=total_seq_len,
        # A separate corpus needs its own cache root, or its file indices would
        # collide with the training set's under on_generate="cache".
        transfer=(val_transfer or transfer) if use_val_path else transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        train_ratio=1.0 if use_val_path else train_data_ratio,
        split="train" if use_val_path else "val",
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
        no_packing=no_packing,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        # Validation must use the SAME batching convention as training: under
        # no_packing the accept-length metric is measured on single-conversation
        # windows, not on packed multi-document ones.
        no_packing=no_packing,
    )

    return train_loader, val_loader
