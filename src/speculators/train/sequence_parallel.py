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
# GQA head replication
# ---------------------------------------------------------------------------


def maybe_replicate_kv_heads(
    key: torch.Tensor,
    value: torch.Tensor,
    sp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate KV heads when ``sp_size > num_kv_heads`` for GQA models.

    When ``sp_size <= num_kv_heads``, validates divisibility and returns
    K/V unchanged.  When ``sp_size > num_kv_heads``, replicates each KV
    head so the total becomes divisible by ``sp_size``, allowing the
    all-to-all scatter to split heads evenly across SP ranks.

    Args:
        key: ``(B, num_kv_heads, S, D)``
        value: ``(B, num_kv_heads, S, D)``
        sp_size: Sequence-parallel world size.
    """
    num_kv_heads = key.shape[1]

    if sp_size <= num_kv_heads:
        if num_kv_heads % sp_size != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"sp_size ({sp_size})"
            )
        return key, value

    if sp_size % num_kv_heads != 0:
        raise ValueError(
            f"sp_size ({sp_size}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) when sp_size > num_kv_heads"
        )

    replication_factor = sp_size // num_kv_heads
    key = key.repeat_interleave(replication_factor, dim=1)
    value = value.repeat_interleave(replication_factor, dim=1)
    return key, value


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


# ---------------------------------------------------------------------------
# DFlash-specific Ulysses attention
# ---------------------------------------------------------------------------


def dflash_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Ulysses SP attention for DFlash with asymmetric Q/KV.

    DFlash K/V are ``[context | noise]`` concatenated along the sequence
    dim.  A naive all-to-all would interleave context and noise from
    different ranks, breaking the mask.  This function splits K/V at
    ``ctx_len``, does separate all-to-alls, and concatenates
    ``[global_ctx | global_noise]`` to preserve the layout that
    ``create_anchor_block_mask_mod`` expects.

    Falls back to the base ``flex_attention_forward`` when SP is
    inactive (``sp_size <= 1``).
    """
    from speculators.train.distributed import get_sp_group, get_sp_size  # noqa: PLC0415

    sp_size = get_sp_size()

    if sp_size <= 1:
        from speculators.models.attention import (  # noqa: PLC0415
            flex_attention_forward,
        )

        return flex_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )

    from torch.nn.attention.flex_attention import flex_attention  # noqa: PLC0415

    ctx_len = kwargs.pop("ctx_len", None)
    if ctx_len is None:
        raise ValueError("ctx_len must be provided for DFlash Ulysses attention")

    sp_group = get_sp_group()
    key, value = maybe_replicate_kv_heads(key, value, sp_size)

    k_ctx, k_noise = key.split([ctx_len, key.shape[2] - ctx_len], dim=2)
    v_ctx, v_noise = value.split([ctx_len, value.shape[2] - ctx_len], dim=2)

    query = _AllToAllSP.apply(query, sp_group, 1, 2)
    k_ctx = _AllToAllSP.apply(k_ctx, sp_group, 1, 2)
    k_noise = _AllToAllSP.apply(k_noise, sp_group, 1, 2)
    v_ctx = _AllToAllSP.apply(v_ctx, sp_group, 1, 2)
    v_noise = _AllToAllSP.apply(v_noise, sp_group, 1, 2)

    key = torch.cat([k_ctx, k_noise], dim=2)
    value = torch.cat([v_ctx, v_noise], dim=2)

    num_q_local = query.shape[1]
    num_kv_local = key.shape[1]

    attn_output = flex_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=num_q_local != num_kv_local,
        scale=scaling,
    )

    attn_output = _AllToAllSP.apply(attn_output, sp_group, 2, 1)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
