"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

import importlib.util
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
from torch.nn.attention.flex_attention import (
    create_mask as _create_mask,
)
from transformers.modeling_utils import AttentionInterface

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def fa4_is_available() -> bool:
    """Check whether the FA4 (FlashAttention-4) backend can be used.

    Requires both Hopper+ GPU (compute capability >= 9.0) and the
    ``flash_attn.cute`` CuTeDSL kernels to be installed.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (9, 0):
        return False
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_fa4_mode: Literal["auto", "on", "off"] = "auto"


def configure_fa4(mode: Literal["auto", "on", "off"] = "auto") -> None:
    """Set the FA4 backend policy.

    * ``"auto"`` (default): use FA4 when :func:`fa4_is_available` is True.
    * ``"on"``: always request FA4 (errors at compile time if unavailable).
    * ``"off"``: never use FA4, always use the default Triton backend.
    """
    global _fa4_mode  # noqa: PLW0603
    _fa4_mode = mode
    enabled = mode == "on" or (mode == "auto" and fa4_is_available())
    logger.info("FA4 flex-attention backend: mode=%s, enabled=%s", mode, enabled)


def _should_use_fa4() -> bool:
    if _fa4_mode == "on":
        return True
    if _fa4_mode == "off":
        return False
    return fa4_is_available()


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

    kernel_options = {"BACKEND": "FLASH"} if _should_use_fa4() else None

    flex_attention_output = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,  # type: ignore[arg-type]
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
