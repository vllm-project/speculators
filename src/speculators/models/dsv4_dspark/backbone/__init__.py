"""Backend-agnostic DSV4 decoder backbone for the DSpark draft.

A self-contained torch implementation of the DeepSeek-V4-Flash decoder shape:
multi-head latent attention with per-head sinks, 256 routed experts + 1 shared, and
hyper-connections. Plain PyTorch throughout, with no dependency on any accelerator
package.
"""

from __future__ import annotations

from .norm import RMSNorm, UnweightedRMSNorm

__all__ = [
    "RMSNorm",
    "UnweightedRMSNorm",
]
