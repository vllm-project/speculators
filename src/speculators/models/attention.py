"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

import json
import os
from collections.abc import Callable
from typing import cast

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)
from torch.nn.attention.flex_attention import (
    create_mask as _create_mask,
)
from transformers.modeling_utils import AttentionInterface

# ~99 KB per block on sm_120 (RTX PRO 6000 / RTX 5090) and sm_89 (L40S / 4090);
# inductor's default flex backward tiles for head_dim > 128 need ~112 KB.
_LOW_SHARED_MEMORY_BYTES = 128 * 1024
_MAX_DEFAULT_TILE_HEAD_DIM = 128
_LOW_SHARED_MEMORY_KERNEL_OPTIONS = {
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
    "bwd_num_stages": 1,
}
# JSON dict, e.g. '{"bwd_BLOCK_M1": 16}'. Parsed once at import so the compiled
# forward only sees a constant: reading os.environ inside the traced function
# causes a dynamo graph break, which drops flex_attention out of the compiled
# graph and back to the dense-materializing eager path.
_ENV_KERNEL_OPTIONS = json.loads(
    os.environ.get("SPECULATORS_FLEX_KERNEL_OPTIONS", "null")
)


def flex_kernel_options(
    device: torch.device, head_dim: int
) -> FlexKernelOptions | None:
    """Flex ``kernel_options`` for this device, or None for inductor defaults.

    The low-shared-memory defaults assume bf16/fp16 training; fp32 needs
    smaller tiles still (e.g. 16x16), which can be forced via the
    ``SPECULATORS_FLEX_KERNEL_OPTIONS`` env var (a JSON dict,
    e.g. ``'{"bwd_BLOCK_M1": 16}'``) set before this module is imported.
    """
    if _ENV_KERNEL_OPTIONS is not None:
        return cast("FlexKernelOptions", dict(_ENV_KERNEL_OPTIONS))
    if device.type != "cuda" or head_dim <= _MAX_DEFAULT_TILE_HEAD_DIM:
        return None
    props = torch.cuda.get_device_properties(device)
    # The attribute exists at runtime; torch's stubs only know the
    # smaller static `shared_memory_per_block`.
    optin = props.shared_memory_per_block_optin  # type: ignore[attr-defined]
    if optin < _LOW_SHARED_MEMORY_BYTES:
        return cast("FlexKernelOptions", dict(_LOW_SHARED_MEMORY_KERNEL_OPTIONS))
    return None


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

    flex_attention_output = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=flex_kernel_options(query.device, query.shape[-1]),
    )
    attention_output: torch.Tensor = flex_attention_output
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
