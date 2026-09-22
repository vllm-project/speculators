"""Utilities for FP8 scaled quantization of hidden states.

Uses per-token scaling: one float32 scale per token, computed over all
non-token dimensions (e.g. ``num_layers`` and ``hidden_size`` combined for a
``[seq_len, num_layers, hidden_size]`` hidden-states tensor).

Quantize: scale = amax / FP8_MAX; fp8 = (tensor / scale).to(fp8_dtype)
Dequantize: restored = fp8.to(target_dtype) * scale
"""

from __future__ import annotations

import torch

SCALES_KEY = "hidden_states_scales"


def _fp8_dtype() -> torch.dtype:
    """Resolve the FP8 dtype lazily (not all torch builds expose it)."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError(
            "This torch build does not expose torch.float8_e4m3fn; "
            "FP8 hidden-states quantization is unavailable."
        )
    return dtype


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with a single scale per token (dim 0).

    Args:
        tensor: Input tensor whose first dimension is the token/sequence
            dimension, e.g. ``[seq_len, num_layers, hidden_size]``, in any
            float dtype.

    Returns:
        Tuple of ``(fp8_tensor, scale)`` where ``scale`` has the same rank as
        ``tensor`` but every dimension except dim 0 is 1 (e.g.
        ``[seq_len, 1, 1]``), so it broadcasts directly against ``tensor``.
    """
    fp8_dtype = _fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    fp32 = tensor.float()
    reduce_dims = tuple(range(1, fp32.dim()))
    amax = fp32.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (amax / fp8_max).clamp(min=1e-12)
    fp8_tensor = (fp32 / scale).to(fp8_dtype)
    return fp8_tensor, scale


def dequantize_fp8_tensor(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor (with its per-token scale) back to ``dtype``."""
    return fp8_tensor.to(dtype) * scale.to(dtype)
