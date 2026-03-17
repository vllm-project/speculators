"""FastMTP dataset utilities for loading pre-generated ``.pt`` training files.

Each ``.pt`` file holds a single sequence in the format written by
:func:`~speculators.data_generation.fast_mtp_generator.generate_and_save_fast_mtp`::

    {
        "input_ids":     Tensor[seq_len],     # long
        "hidden_states": Tensor[seq_len, H],  # float32, last verifier layer
        "loss_mask":     Tensor[seq_len],     # long, 1 = compute loss here
    }

:class:`FastMTPSampleFileDataset` loads these files, applies the alignment
shift needed by :class:`~speculators.models.fast_mtp.FastMTPSpeculator`, and
is compatible with the :func:`~speculators.train.data.create_collate_fn`
packing used by Eagle3.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from speculators.train.data import (
    create_collate_fn,
    list_files,
    split_files,
)

__all__ = [
    "FastMTPSampleFileDataset",
    "make_fast_mtp_dataloader",
]

BatchType = dict[str, Any]


def _shift_batch_fastmtp(batch: BatchType) -> BatchType:
    """Align input_ids and hidden_states for FastMTPSpeculator.forward.

    ``FastMTPSpeculator.forward`` at step *k* uses:

    * ``embed(input_ids[:, k+1 : k+1+L])``  — token embedding
    * ``hidden_states[:, :L]``              — verifier hidden state
    * Labels: ``input_ids[:, k+2 : k+2+L]``

    After this shift ``input_ids[n]`` = ``x_{n+1}`` and
    ``hidden_states[n]`` = ``g_n``, so the model at step 0 reads
    ``embed(x_2, …)`` from position 1 onward — correct alignment for
    predicting ``x_3, …``.

    Reduces ``seq_len`` by 1 (same as Eagle3's ``shift_batch``).
    """
    return {
        "input_ids": batch["input_ids"][1:],
        "hidden_states": batch["hidden_states"][:-1],
        "loss_mask": batch["loss_mask"][1:],
        "lengths": batch["lengths"] - 1,
        "position_ids": batch["position_ids"][1:],
    }


class FastMTPSampleFileDataset(Dataset):
    """Dataset that loads FastMTP ``.pt`` files written by the data generator.

    Compatible with :func:`~speculators.train.data.create_collate_fn` —
    returns the same key set as ``Eagle3SampleFileDataset`` minus
    ``verifier_last_hidden_states``.

    :param max_len: Maximum sequence length after shift (for collation).
    :param datapath: Root directory; all ``.pt`` files found recursively.
        Mutually exclusive with ``file_list``.
    :param file_list: Explicit list of ``.pt`` paths.
        Mutually exclusive with ``datapath``.
    :param hidden_states_dtype: Cast hidden states to this dtype on load.
    """

    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        hidden_states_dtype: torch.dtype = torch.float32,
    ) -> None:
        if datapath is not None and file_list is not None:
            raise ValueError("Provide datapath or file_list, not both.")
        if datapath is not None:
            file_list = list_files(datapath)
        if file_list is None:
            raise ValueError("Either datapath or file_list must be provided.")

        self.data = file_list
        self.max_len = max_len
        self.hidden_states_dtype = hidden_states_dtype

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> BatchType:
        data = torch.load(
            self.data[index], mmap=True, weights_only=True, map_location="cpu"
        )
        data["hidden_states"] = data["hidden_states"].to(self.hidden_states_dtype)

        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)

        return _shift_batch_fastmtp(data)


def make_fast_mtp_dataloader(
    data_dir: str | Path,
    max_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    hidden_states_dtype: torch.dtype = torch.bfloat16,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation ``DataLoader``s from a FastMTP data directory.

    :param data_dir: Directory containing ``.pt`` files (searched recursively).
    :param max_len: Collation target length; sequences are sliced/padded here.
    :param batch_size: Number of samples per batch.
    :param train_ratio: Fraction of files used for training (rest = validation).
    :param hidden_states_dtype: dtype for hidden states tensors.
    :return: ``(train_loader, val_loader)``
    """
    train_files, val_files = split_files(str(data_dir), ratio=train_ratio)
    collate_fn = create_collate_fn(max_len)

    def _make_loader(files: list[str], shuffle: bool) -> DataLoader:
        ds = FastMTPSampleFileDataset(
            max_len=max_len,
            file_list=files,
            hidden_states_dtype=hidden_states_dtype,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )

    return (
        _make_loader([str(f) for f in train_files], shuffle=True),
        _make_loader([str(f) for f in val_files], shuffle=False),
    )
