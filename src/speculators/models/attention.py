"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

from collections.abc import Callable

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
from torch.nn.attention.flex_attention import (
    create_mask as _create_mask,
)
from transformers.modeling_utils import AttentionInterface


def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Shared flex attention forward with optional Ulysses sequence parallelism.

    When ``sp_size > 1``, Q/K/V are transposed via all-to-all from
    sequence-parallel layout ``(B, H, S_local, D)`` to head-parallel
    layout ``(B, H/sp, S_full, D)`` before attention, and the output
    is transposed back afterwards.

    Args:
        module: The attention module (unused but required for interface compatibility).
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim).
        attention_mask: BlockMask for flex attention.
        scaling: Optional scaling factor for attention scores.
        **_kwargs: Additional unused kwargs for interface compatibility.

    Returns:
        Tuple of (attention_output, None) where attention_output has shape
        (batch, seq_len, num_heads, head_dim) and None represents no attention weights.
    """
    from speculators.train.distributed import get_sp_group, get_sp_size  # noqa: PLC0415
    from speculators.train.sequence_parallel import (  # noqa: PLC0415
        ulysses_gather,
        ulysses_scatter,
    )

    sp_size = get_sp_size()
    use_sp = sp_size > 1

    if use_sp:
        sp_group = get_sp_group()
        query = ulysses_scatter(query, sp_group, sp_size)
        key = ulysses_scatter(key, sp_group, sp_size)
        value = ulysses_scatter(value, sp_group, sp_size)

    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    enable_gqa = num_query_heads != num_key_value_heads

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    flex_attention_output = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
    )
    attention_output: torch.Tensor = flex_attention_output

    if use_sp:
        attention_output = ulysses_gather(attention_output, sp_group, sp_size)

    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


def create_float_mask(
    mask_mod: Callable,
    B: int | None = None,  # noqa: N803
    H: int | None = None,  # noqa: N803
    Q_LEN: int = 0,  # noqa: N803
    KV_LEN: int = 0,  # noqa: N803
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Wrap ``create_mask`` and convert the boolean result to a float mask.

    Non-flex attention backends (eager, SDPA) add the mask numerically
    (``scores + mask``) and need 0 for attended and ``-inf`` for masked.
    """
    bool_mask = _create_mask(
        mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )
    float_mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    float_mask.masked_fill_(~bool_mask, float("-inf"))
    return float_mask


def block_mask_to_dense_attention_mask(
    block_mask: BlockMask, device: torch.device, dtype: torch.dtype
):
    attention_mask = torch.ones(block_mask.shape, device=device, dtype=dtype)

    for q_idx in range(attention_mask.shape[2]):
        attention_mask[0, 0, q_idx, :] = block_mask.mask_mod(
            torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.ones(1, device=device, dtype=torch.long) * q_idx,
            torch.arange(attention_mask.shape[3], device=device, dtype=torch.long),
        )
    return attention_mask


# Singleton registry for attention functions (shared across all models)
ALL_ATTENTION_FUNCTIONS = AttentionInterface()
ALL_ATTENTION_FUNCTIONS.register("simple_flex_attention", flex_attention_forward)
