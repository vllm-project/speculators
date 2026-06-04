"""Patch HF rotary helper to match vLLM partial-neox behavior.

HF ``apply_rotary_pos_emb`` rotates by splitting at ``head_dim/2``.
vLLM partial MRoPE rotates only the first ``rotary_dim`` channels and
keeps the tail unchanged. This file aligns HF training with that runtime
behavior, while keeping full-rotation paths unchanged.
"""

from __future__ import annotations

import torch

__all__ = [
    "install_partial_neox_rotary",
    "partial_neox_apply_rotary_pos_emb",
]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF/neox "rotate_half" — splits the last dim in half and swaps."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def partial_neox_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-compatible rotary helper with partial-neox fallback.

    - If ``cos`` covers full head dim, behavior matches HF.
    - If ``cos`` is shorter, rotate only the first ``rotary_dim`` channels
      and keep the remaining channels unchanged.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if cos.shape[-1] == q.shape[-1]:
        # Full rotation — identical to HF apply_rotary_pos_emb.
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    if cos.shape[-1] > q.shape[-1]:
        raise ValueError(
            f"cos last dim ({cos.shape[-1]}) exceeds q last dim "
            f"({q.shape[-1]}); rotary tables larger than head_dim are "
            "unsupported by this partial-neox replacement."
        )

    rotary_dim = cos.shape[-1]
    # Rotate leading rotary channels; keep tail as pass-through.
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


_INSTALLED = False


def install_partial_neox_rotary() -> None:
    """Patch ``apply_rotary_pos_emb`` in HF ``llama`` and ``qwen3`` modules.

    Idempotent. Full-rotation paths keep original behavior.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    # Local imports — keep transformers a soft dep at module import time.
    from transformers.models.llama import modeling_llama  # noqa: PLC0415
    from transformers.models.qwen3 import modeling_qwen3  # noqa: PLC0415

    for module in (modeling_llama, modeling_qwen3):
        original = module.apply_rotary_pos_emb
        # Cache original for tests / debugging — and to allow uninstall.
        if not hasattr(module, "_speculators_original_apply_rotary_pos_emb"):
            module._speculators_original_apply_rotary_pos_emb = original  # type: ignore[attr-defined]
        module.apply_rotary_pos_emb = partial_neox_apply_rotary_pos_emb

    _INSTALLED = True


def uninstall_partial_neox_rotary() -> None:
    """Restore HF's original ``apply_rotary_pos_emb``. Test/debug helper."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from transformers.models.llama import modeling_llama  # noqa: PLC0415
    from transformers.models.qwen3 import modeling_qwen3  # noqa: PLC0415

    for module in (modeling_llama, modeling_qwen3):
        original = getattr(
            module, "_speculators_original_apply_rotary_pos_emb", None
        )
        if original is not None:
            module.apply_rotary_pos_emb = original
    _INSTALLED = False
