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


# ---------------------------------------------------------------------------
# Differentiable all-to-all primitive
# ---------------------------------------------------------------------------


class _AllToAllSP(torch.autograd.Function):
    """Differentiable all-to-all for sequence parallelism.

    Scatters along ``scatter_dim`` and gathers along ``gather_dim``.
    The backward pass reverses the two dimensions so gradients flow
    back through the communication correctly.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        sp_group: ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = dist.get_world_size(sp_group)
        if world_size == 1:
            return input_tensor

        input_chunks = input_tensor.chunk(world_size, dim=scatter_dim)
        output_chunks = [torch.empty_like(c) for c in input_chunks]
        dist.all_to_all(output_chunks, list(input_chunks), group=sp_group)
        return torch.cat(output_chunks, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _AllToAllSP.apply(
                grad_output, ctx.sp_group, ctx.gather_dim, ctx.scatter_dim
            ),
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Ulysses scatter / gather for attention
# ---------------------------------------------------------------------------


def ulysses_scatter(
    x: torch.Tensor,
    sp_group: ProcessGroup,
    sp_size: int,  # noqa: ARG001
) -> torch.Tensor:
    """All-to-all: sequence-parallel to head-parallel layout.

    Input:  ``(B, H, S_local, D)``
    Output: ``(B, H // sp_size, S_full, D)``

    Scatters heads (dim 1) and gathers sequence chunks (dim 2).
    Gradients flow back through the reverse all-to-all automatically.
    """
    return _AllToAllSP.apply(x, sp_group, 1, 2)


def ulysses_gather(
    x: torch.Tensor,
    sp_group: ProcessGroup,
    sp_size: int,  # noqa: ARG001
) -> torch.Tensor:
    """All-to-all: head-parallel to sequence-parallel layout.

    Input:  ``(B, H_local, S_full, D)``
    Output: ``(B, H_local * sp_size, S_full // sp_size, D)``

    Inverse of :func:`ulysses_scatter`.
    """
    return _AllToAllSP.apply(x, sp_group, 2, 1)
