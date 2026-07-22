"""Ulysses sequence parallelism utilities.

Provides batch splitting and all-to-all communication primitives
for distributing packed sequences across SP ranks.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

_SPLIT_KEYS = frozenset(
    {
        "hidden_states",
        "verifier_last_hidden_states",
        "input_ids",
        "loss_mask",
        "position_ids",
    }
)

_KEEP_FULL_KEYS = frozenset({"document_ids"})

_MIN_SPLIT_NDIM = 2


def split_batch_for_sp(
    batch: dict[str, torch.Tensor],
    sp_rank: int,
    sp_size: int,
) -> dict[str, torch.Tensor]:
    """Split batch tensors along the sequence dimension for SP.

    Tensors in ``_SPLIT_KEYS`` are chunked along dim 1 and the chunk
    for *sp_rank* is kept.  ``document_ids`` is kept full (needed for
    attention mask construction over the global sequence).

    Adds ``full_seq_len`` to the returned dict so downstream code
    can recover the original sequence length.
    """
    if sp_size <= 1:
        return batch

    out: dict[str, torch.Tensor] = {}
    full_seq_len: int | None = None

    for key, tensor in batch.items():
        if key in _SPLIT_KEYS and tensor.dim() >= _MIN_SPLIT_NDIM:
            seq_len = tensor.shape[1]
            if full_seq_len is None:
                full_seq_len = seq_len
            chunks = tensor.chunk(sp_size, dim=1)
            out[key] = chunks[sp_rank].contiguous()
        elif key in _KEEP_FULL_KEYS:
            if full_seq_len is None and tensor.dim() >= _MIN_SPLIT_NDIM:
                full_seq_len = tensor.shape[1]
            out[key] = tensor
        else:
            out[key] = tensor

    if full_seq_len is not None:
        out["full_seq_len"] = torch.tensor(full_seq_len, device=tensor.device)

    return out


def ulysses_scatter(
    x: torch.Tensor,
    sp_group: ProcessGroup,
    sp_size: int,
) -> torch.Tensor:
    """All-to-all: sequence-parallel to head-parallel layout.

    Input:  ``(B, H, S_local, D)``
    Output: ``(B, H // sp_size, S_full, D)``

    Each rank scatters ``sp_size`` head-chunks and gathers ``sp_size``
    sequence-chunks, yielding the full sequence over fewer heads.
    """
    bsz, num_heads, seq_local, head_dim = x.shape
    heads_local = num_heads // sp_size

    x = x.reshape(bsz, sp_size, heads_local, seq_local, head_dim)
    input_list = [x[:, i].contiguous() for i in range(sp_size)]
    output_list = [
        torch.empty_like(input_list[0]) for _ in range(sp_size)
    ]
    dist.all_to_all(output_list, input_list, group=sp_group)
    return torch.cat(output_list, dim=2)


def ulysses_gather(
    x: torch.Tensor,
    sp_group: ProcessGroup,
    sp_size: int,
) -> torch.Tensor:
    """All-to-all: head-parallel to sequence-parallel layout.

    Input:  ``(B, H_local, S_full, D)``
    Output: ``(B, H_local * sp_size, S_full // sp_size, D)``

    Inverse of :func:`ulysses_scatter`.
    """
    bsz, heads_local, seq_full, head_dim = x.shape
    seq_local = seq_full // sp_size

    x = x.reshape(bsz, heads_local, sp_size, seq_local, head_dim)
    input_list = [x[:, :, i].contiguous() for i in range(sp_size)]
    output_list = [
        torch.empty_like(input_list[0]) for _ in range(sp_size)
    ]
    dist.all_to_all(output_list, input_list, group=sp_group)
    return torch.cat(output_list, dim=1)
