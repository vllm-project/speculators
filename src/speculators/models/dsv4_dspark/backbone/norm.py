"""RMS normalization variants for the DSV4 DSpark draft backbone.

Two flavors, both fp32-accumulated for bf16 stability:

* :class:`RMSNorm` — learnable per-channel scale, ones-init: ``y = w · x̂``.
* :class:`UnweightedRMSNorm` — no parameters: ``y = x̂`` (the scale is folded
  into a downstream projection). Used on the per-head query inside MLA.

where ``x̂ = x / sqrt(mean(x², -1) + eps)``.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Ones-init RMSNorm with fp32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        x_normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_normed.to(in_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class UnweightedRMSNorm(nn.Module):
    """Parameter-free RMS normalization (scale folded downstream)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
            x.dtype
        )

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
