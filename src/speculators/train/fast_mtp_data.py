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

import random
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

from speculators.train.data import create_collate_fn

__all__ = [
    "FastMTPBatch",
    "FastMTPSampleFileDataset",
    "make_fast_mtp_dataloader",
]


class FastMTPBatch(TypedDict):
    """Single post-shift training sample returned by :class:`FastMTPSampleFileDataset`.

    All tensors are 1-D (unbatched); the DataLoader collates them into 2-D batches.
    ``seq_len`` is the original sequence length minus 1 (from the shift).
    """

    input_ids: torch.Tensor      # [seq_len]     — long
    hidden_states: torch.Tensor  # [seq_len, H]  — float (dtype from dataset config)
    loss_mask: torch.Tensor      # [seq_len]     — long, 1 = compute loss
    lengths: torch.Tensor        # [1]           — long, equals seq_len
    position_ids: torch.Tensor   # [seq_len]     — long, 0…seq_len-1


def _shift_batch_fastmtp(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    loss_mask: torch.Tensor,
) -> FastMTPBatch:
    """Align input_ids and hidden_states for FastMTPSpeculator.forward.

    The forward pass at step *k*, position *t* uses:

    * ``embed(input_ids[t+k])``   — token embedding (shifted 1 position forward)
    * ``hidden_states[t]``        — verifier hidden state (unshifted within each step)
    * Labels: ``input_ids[t+k+1]``

    After this shift ``input_ids[n] = x_{n+1}`` and ``hidden_states[n] = g_n``,
    so at step 0 the model reads ``embed(x_1, x_2, …)`` from position 0 onward and
    predicts ``x_2, x_3, …``.

    Reduces ``seq_len`` by 1 (same as Eagle3's ``shift_batch``).

    :param input_ids: Token IDs [seq_len] — long.
    :param hidden_states: Verifier hidden states [seq_len, H].
    :param loss_mask: Binary loss mask [seq_len] — long.
    :returns: Shifted :class:`FastMTPBatch` with ``seq_len - 1`` valid positions.
    """
    shifted_input_ids = input_ids[1:]
    shifted_len = shifted_input_ids.shape[0]

    return FastMTPBatch(
        input_ids=shifted_input_ids,
        hidden_states=hidden_states[:-1],
        loss_mask=loss_mask[1:],
        lengths=torch.tensor([shifted_len], dtype=torch.long),
        position_ids=torch.arange(shifted_len, dtype=torch.long),
    )


class FastMTPSampleFileDataset(Dataset):
    """Dataset that loads FastMTP ``.pt`` files written by the data generator.

    Compatible with :func:`~speculators.train.data.create_collate_fn` —
    returns the same key set as ``Eagle3SampleFileDataset`` minus
    ``verifier_last_hidden_states``.

    :param max_len: Maximum sequence length after shift (for collation).
    :param datapath: Root directory; all ``data_*.pt`` files found recursively.
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
            file_list = [str(f) for f in sorted(Path(datapath).glob("data_*.pt"))]
        if file_list is None:
            raise ValueError("Either datapath or file_list must be provided.")

        self.file_paths = file_list
        self.max_len = max_len
        self.hidden_states_dtype = hidden_states_dtype

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> FastMTPBatch:
        path = self.file_paths[index]
        try:
            data = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load sample at {path!r}") from exc

        missing = {"input_ids", "hidden_states", "loss_mask"} - data.keys()
        if missing:
            raise KeyError(f"Sample {path!r} is missing required keys: {missing}")

        data["hidden_states"] = data["hidden_states"].to(self.hidden_states_dtype)

        return _shift_batch_fastmtp(
            input_ids=data["input_ids"],
            hidden_states=data["hidden_states"],
            loss_mask=data["loss_mask"],
        )


def make_fast_mtp_dataloader(
    data_dir: str | Path,
    max_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    hidden_states_dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation ``DataLoader``s from a FastMTP data directory.

    :param data_dir: Directory containing ``data_*.pt`` files.
    :param max_len: Collation target length; sequences are sliced/padded here.
    :param batch_size: Number of samples per batch.
    :param train_ratio: Fraction of files used for training (rest = validation).
    :param hidden_states_dtype: dtype for hidden states tensors.
    :param seed: Random seed for the train/val shuffle split.
    :return: ``(train_loader, val_loader)``
    """
    all_files = sorted(Path(data_dir).glob("data_*.pt"))
    random.seed(seed)
    random.shuffle(all_files)
    split = int(len(all_files) * train_ratio)
    train_files, val_files = all_files[:split], all_files[split:]
    collate_fn = create_collate_fn(max_len)

    def _make_loader(files: list, shuffle: bool) -> DataLoader:
        ds = FastMTPSampleFileDataset(
            max_len=max_len,
            file_list=[str(f) for f in files],
            hidden_states_dtype=hidden_states_dtype,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )

    return _make_loader(train_files, shuffle=True), _make_loader(
        val_files, shuffle=False
    )
