"""Utilities for FP8 scaled quantization of hidden states.

Uses per-token scaling: one float32 scale per token, computed over all
non-token dimensions (e.g. ``num_layers`` and ``hidden_size`` combined for a
``[seq_len, num_layers, hidden_size]`` hidden-states tensor).

Quantize: scale = amax / FP8_MAX; fp8 = clamp(tensor / scale, ±FP8_MAX).to(fp8)
Dequantize: restored = fp8.to(target_dtype) * scale
"""

from __future__ import annotations

import torch

from hs_connectors.mooncake_store import assert_finite

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

    Raises:
        NonFiniteTensorError: If ``tensor`` is a non-empty floating tensor
            containing NaN or infinity, or if a finite value is too large for
            the float32 per-token scale. Nothing is quantized in that case.
    """
    # Same gate Mooncake uses before publish: do not quantize NaN or Inf.
    assert_finite("hidden_states", tensor)
    fp8_dtype = _fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    # A finite float64 value above the float32 range becomes Inf in
    # ``tensor.float()``. Scaling after that is Inf/Inf and the clamp cannot
    # turn the resulting NaN into an FP8 value. Keep float64 until the scale
    # exists. Every other dtype already fits in float32.
    work = tensor if tensor.dtype == torch.float64 else tensor.float()
    reduce_dims = tuple(range(1, work.dim()))
    amax = work.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (amax / fp8_max).clamp(min=1e-12)
    # e4m3 has no infinity. amax / scale can round slightly above fp8_max,
    # and that overflow becomes NaN instead of saturating.
    scaled = (work / scale).clamp(-fp8_max, fp8_max)
    if work.dtype == torch.float64:
        # The connector stores this scale as float32. Past float32 max * fp8
        # max the scale itself is Inf, which is still a non-finite write.
        scale = scale.float()
        scaled = scaled.float()
        assert_finite("hidden_states", scale)
    fp8_tensor = scaled.to(fp8_dtype)
    return fp8_tensor, scale


def dequantize_fp8_tensor(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor (with its per-token scale) back to ``dtype``."""
    return fp8_tensor.to(dtype) * scale.to(dtype)
