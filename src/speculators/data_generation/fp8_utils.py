"""Utilities for FP8 scaled quantization of hidden states.

Uses per-token scaling: one float32 scale per row/token (shape [seq_len, 1]).

Quantize: scale = amax / FP8_MAX; fp8 = (tensor / scale).to(fp8_dtype)
Dequantize: restored = fp8.to(target_dtype) * scale
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0
SCALES_KEY = "hidden_states_scales"


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single tensor to FP8 with per-token scaling.

    Args:
        tensor: Input tensor of shape [seq_len, hidden_size] in any float dtype.

    Returns:
        Tuple of (fp8_tensor, scale) where scale shape is [seq_len, 1].
    """
    fp32 = tensor.float()
    amax = fp32.abs().amax(dim=-1, keepdim=True)  # [seq_len, 1]
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    fp8_tensor = (fp32 / scale).to(FP8_DTYPE)
    return fp8_tensor, scale


def dequantize_fp8_tensor(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to the target dtype."""
    return fp8_tensor.to(dtype) * scale.to(dtype)
