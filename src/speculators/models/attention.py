"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

from typing import cast

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.modeling_utils import AttentionInterface

_compiled_flex_attention = torch.compile(flex_attention)


def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Shared flex attention forward implementation.

    This function is used by both EAGLE3 and DFlash attention mechanisms to avoid
    code duplication and ensure consistent behavior.

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
    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    enable_gqa = num_query_heads != num_key_value_heads

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    flex_attention_output = _compiled_flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
    )
    attention_output: torch.Tensor = cast("torch.Tensor", flex_attention_output)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


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
