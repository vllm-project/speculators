"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
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
    """Shared flex attention forward implementation.

    Falls back to scaled_dot_product_attention on devices that don't support
    flex_attention (e.g. NPU/Ascend).

    Args:
        module: The attention module (unused but required for interface compatibility).
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim).
        attention_mask: BlockMask for flex attention, or a dense float mask.
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

    # FlexAttention is only supported on CUDA, CPU, and HPU.
    # Fall back to SDPA on NPU and other devices.
    if query.device.type in ("cuda", "cpu", "hpu"):
        flex_attention_output = flex_attention(
            query,
            key,
            value,
            score_mod=None,
            block_mask=attention_mask,
            enable_gqa=enable_gqa,
            scale=scaling,
        )
        attention_output = flex_attention_output.transpose(1, 2).contiguous()
        return attention_output, None

    # NPU / unsupported device fallback: use scaled_dot_product_attention
    if isinstance(attention_mask, BlockMask):
        # Materialize BlockMask into a dense float mask
        dense_mask = torch.ones(
            (attention_mask.shape[0], 1, attention_mask.shape[2], attention_mask.shape[3]),
            device=query.device,
            dtype=query.dtype,
        )
        for q_idx in range(attention_mask.shape[2]):
            dense_mask[0, 0, q_idx, :] = attention_mask.mask_mod(
                torch.zeros(1, device=query.device, dtype=torch.long),
                torch.zeros(1, device=query.device, dtype=torch.long),
                torch.ones(1, device=query.device, dtype=torch.long) * q_idx,
                torch.arange(attention_mask.shape[3], device=query.device, dtype=torch.long),
            )
        # Convert to bool mask (True = keep, False = mask)
        attn_mask = dense_mask > 0.5
    elif isinstance(attention_mask, torch.Tensor):
        if attention_mask.dtype == torch.bool:
            attn_mask = attention_mask
        else:
            # Float mask: convert to bool (0.0 = attend, negative = mask)
            attn_mask = attention_mask > -1.0
    elif attention_mask is None:
        attn_mask = None
    else:
        attn_mask = None

    # Repeat GQA key/value heads to match query heads
    if enable_gqa:
        key = key.repeat_interleave(num_query_heads // num_key_value_heads, dim=1)
        value = value.repeat_interleave(num_query_heads // num_key_value_heads, dim=1)

    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        scale=scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


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
