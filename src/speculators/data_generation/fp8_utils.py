"""Utilities for FP8 scaled quantization of hidden states.

Supports two granularities:
  - per_tensor_scaled: one float32 scale per [seq_len, hidden_size] tensor (shape [1])
  - per_token_scaled:  one float32 scale per row/token (shape [seq_len, 1])

Quantize: scale = amax / FP8_MAX; fp8 = (tensor / scale).to(fp8_dtype)
Dequantize: restored = fp8.to(target_dtype) * scale

Both granularities use the same dequantize path — broadcasting handles the
shape difference transparently.
"""

from __future__ import annotations

from typing import Literal

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0
FP8_FORMAT_KEY = "fp8_format"
FP8_FORMAT_PER_TENSOR = "per_tensor_scaled"
FP8_FORMAT_PER_TOKEN = "per_token_scaled"
SCALES_KEY = "hidden_states_scales"

Granularity = Literal["per_tensor", "per_token"]


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
    granularity: Granularity = "per_token",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single tensor to FP8 with scaling.

    Args:
        tensor: Input tensor of shape [seq_len, hidden_size] in any float dtype.
        granularity: "per_tensor" for one global scale, "per_token" for one
                     scale per row (token position).

    Returns:
        Tuple of (fp8_tensor, scale).
        - per_tensor: scale shape [1]
        - per_token:  scale shape [seq_len, 1]
    """
    fp32 = tensor.float()
    if granularity == "per_token":
        amax = fp32.abs().amax(dim=-1, keepdim=True)  # [seq_len, 1]
    else:
        amax = fp32.abs().amax().reshape(1)  # [1]
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    fp8_tensor = (fp32 / scale).to(FP8_DTYPE)
    return fp8_tensor, scale


def dequantize_fp8_tensor(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to the target dtype.

    Works for both per-tensor (scale shape [1]) and per-token (scale shape
    [seq_len, 1]) — broadcasting handles both cases.
    """
    return fp8_tensor.to(dtype) * scale.to(dtype)


def quantize_hidden_states(
    hidden_states: list[torch.Tensor],
    granularity: Granularity = "per_token",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Quantize a list of hidden state tensors (one per layer) to FP8.

    Args:
        hidden_states: List of tensors, each [seq_len, hidden_size].
        granularity: Quantization granularity.

    Returns:
        Tuple of (fp8_tensors, scales) where each list has one entry per layer.
    """
    fp8_tensors = []
    scales = []
    for h in hidden_states:
        fp8_h, s = quantize_tensor_to_fp8(h, granularity)
        fp8_tensors.append(fp8_h)
        scales.append(s)
    return fp8_tensors, scales


def dequantize_hidden_states(
    fp8_hidden_states: list[torch.Tensor],
    scales: list[torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Dequantize a list of FP8 hidden state tensors back to the target dtype.

    Works for both per-tensor and per-token scales (broadcasting handles both).
    """
    return [
        dequantize_fp8_tensor(h, s, dtype)
        for h, s in zip(fp8_hidden_states, scales)
    ]


def _format_value(granularity: Granularity) -> str:
    if granularity == "per_token":
        return FP8_FORMAT_PER_TOKEN
    return FP8_FORMAT_PER_TENSOR


def pack_fp8_sample(
    input_ids: torch.Tensor,
    hidden_states: list[torch.Tensor],
    loss_mask: torch.Tensor,
    granularity: Granularity = "per_token",
) -> dict:
    """Quantize hidden states and pack into the v2 FP8 data format.

    Args:
        input_ids: Token IDs tensor.
        hidden_states: List of bf16/fp32 hidden state tensors.
        loss_mask: Loss mask tensor.
        granularity: Quantization granularity.

    Returns:
        Dict in v2 format with fp8 hidden states, scales, and format marker.
    """
    fp8_hidden, scales = quantize_hidden_states(hidden_states, granularity)
    return {
        "input_ids": input_ids,
        "hidden_states": fp8_hidden,
        SCALES_KEY: scales,
        "loss_mask": loss_mask,
        FP8_FORMAT_KEY: _format_value(granularity),
    }
