"""Utilities for FP8 scaled quantization of hidden states.

Uses per-token scaling: one float32 scale per row/token (shape [seq_len, 1]).

Quantize: scale = amax / FP8_MAX; fp8 = (tensor / scale).to(fp8_dtype)
Dequantize: restored = fp8.to(target_dtype) * scale
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0
FP8_FORMAT_KEY = "fp8_format"
FP8_FORMAT_VALUE = "per_token_scaled"
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


def quantize_hidden_states(
    hidden_states: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Quantize a list of hidden state tensors (one per layer) to FP8.

    Args:
        hidden_states: List of tensors, each [seq_len, hidden_size].

    Returns:
        Tuple of (fp8_tensors, scales) where each list has one entry per layer.
    """
    fp8_tensors = []
    scales = []
    for h in hidden_states:
        fp8_h, s = quantize_tensor_to_fp8(h)
        fp8_tensors.append(fp8_h)
        scales.append(s)
    return fp8_tensors, scales


def dequantize_hidden_states(
    fp8_hidden_states: list[torch.Tensor],
    scales: list[torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Dequantize a list of FP8 hidden state tensors back to the target dtype."""
    return [
        dequantize_fp8_tensor(h, s, dtype)
        for h, s in zip(fp8_hidden_states, scales)
    ]


def pack_fp8_sample(
    input_ids: torch.Tensor,
    hidden_states: list[torch.Tensor],
    loss_mask: torch.Tensor,
) -> dict:
    """Quantize hidden states and pack into the FP8 data format.

    Args:
        input_ids: Token IDs tensor.
        hidden_states: List of bf16/fp32 hidden state tensors.
        loss_mask: Loss mask tensor.

    Returns:
        Dict with fp8 hidden states, scales, and format marker.
    """
    fp8_hidden, scales = quantize_hidden_states(hidden_states)
    return {
        "input_ids": input_ids,
        "hidden_states": fp8_hidden,
        SCALES_KEY: scales,
        "loss_mask": loss_mask,
        FP8_FORMAT_KEY: FP8_FORMAT_VALUE,
    }
