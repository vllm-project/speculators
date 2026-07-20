"""Explicit module factories for the DFlash Qwen3 backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3MLP, Qwen3RMSNorm

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class DFlashKernels:
    """Construct the replaceable Qwen3 modules used by a DFlash draft."""

    make_rms_norm: Callable[[int, float], nn.Module]
    make_mlp: Callable[[Qwen3Config], nn.Module]


def _make_qwen3_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    return Qwen3RMSNorm(hidden_size, eps=eps)


def _make_qwen3_mlp(config: Qwen3Config) -> nn.Module:
    return Qwen3MLP(config)


DEFAULT_DFLASH_KERNELS = DFlashKernels(
    make_rms_norm=_make_qwen3_rms_norm,
    make_mlp=_make_qwen3_mlp,
)


def load_liger_dflash_kernels() -> DFlashKernels:
    """Load Liger lazily and adapt it to the DFlash construction boundary."""
    try:
        from liger_kernel.transformers import (  # noqa: PLC0415
            LigerRMSNorm,
            LigerSwiGLUMLP,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"liger_kernel", "liger_kernel.transformers"}:
            raise ImportError(
                "--use-liger-kernel requires the optional `speculators[liger]` "
                'extra. Install it with `pip install "speculators[liger]"`.'
            ) from exc
        raise

    def make_rms_norm(hidden_size: int, eps: float) -> nn.Module:
        return LigerRMSNorm(hidden_size, eps=eps)

    def make_mlp(config: Qwen3Config) -> nn.Module:
        return LigerSwiGLUMLP(config)

    return DFlashKernels(
        make_rms_norm=make_rms_norm,
        make_mlp=make_mlp,
    )
