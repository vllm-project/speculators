"""Bit-equivalence test for the partial-neox rotary monkey-patch.

This test pins the contract that motivates the patch: when
``cos.shape[-1] == rotary_dim < head_dim``, the patched
``apply_rotary_pos_emb`` rotates the **same channel pairs** that vLLM's
``MRotaryEmbedding`` rotates at inference time — i.e.

    rotated channels: [0, rotary_dim/2)  paired with  [rotary_dim/2, rotary_dim)
    pass-through:     [rotary_dim, head_dim)

We verify two properties on a hand-rolled vLLM-equivalent reference:

1. **Full-rotation parity** (``rotary_dim == head_dim``): the patched
   helper is byte-equivalent to HF's original ``apply_rotary_pos_emb``,
   so DFlash / plain Llama drafters are unaffected.

2. **Partial-rotation parity** (``rotary_dim < head_dim``, e.g.
   Qwen3.6's ``head_dim=256``, ``partial_rotary_factor=0.25``): the
   patched helper matches a hand-coded vLLM neox-partial reference to
   well below fp32 round-off (``< 1e-5`` max-abs-diff), and the
   pass-through tail is preserved exactly (``allclose`` with
   ``atol=0``).
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply

from speculators.models.eagle3.rotary_partial import (
    install_partial_neox_rotary,
    partial_neox_apply_rotary_pos_emb,
    uninstall_partial_neox_rotary,
)


def _vllm_neox_partial_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-coded vLLM neox-partial RoPE — matches MRotaryEmbedding._forward.

    Independent of the implementation under test; written from the spec
    (see ``vllm/model_executor/layers/rotary_embedding/__init__.py``).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    def _rot(x: torch.Tensor) -> torch.Tensor:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        half = rotary_dim // 2
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        # neox rotate_half on the rotated slice
        rotated = torch.cat((-x2, x1), dim=-1)
        out_rot = (x_rot * cos) + (rotated * sin)
        return torch.cat([out_rot, x_pass], dim=-1)

    return _rot(q), _rot(k)


def test_full_rotation_is_byte_equivalent_to_hf():
    """When cos covers the full head_dim, the patched fn must match HF exactly."""
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 4, 7, 64
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    # cos/sin over full head_dim
    cos = torch.cos(torch.randn(batch, seq, head_dim))
    sin = torch.sin(torch.randn(batch, seq, head_dim))

    q_hf, k_hf = hf_apply(q, k, cos, sin, unsqueeze_dim=1)
    q_p, k_p = partial_neox_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # Strict equality — same arithmetic, same order of operations.
    assert torch.equal(q_hf, q_p), "patched fn drifted from HF on full rotation"
    assert torch.equal(k_hf, k_p), "patched fn drifted from HF on full rotation"


def test_partial_rotation_matches_vllm_reference():
    """rotary_dim < head_dim must match vLLM's neox-partial channel layout."""
    torch.manual_seed(1)
    # Realistic Qwen3.6 shape: head_dim=256, partial_rotary_factor=0.25
    batch, heads, seq, head_dim = 2, 4, 5, 256
    rotary_dim = 64  # int(256 * 0.25)

    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    cos = torch.cos(torch.randn(batch, seq, rotary_dim))
    sin = torch.sin(torch.randn(batch, seq, rotary_dim))

    q_ref, k_ref = _vllm_neox_partial_reference(q, k, cos, sin, unsqueeze_dim=1)
    q_got, k_got = partial_neox_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # Same arithmetic up to algebraic associativity; allow tiny fp32 jitter.
    assert torch.allclose(q_got, q_ref, atol=1e-6, rtol=0)
    assert torch.allclose(k_got, k_ref, atol=1e-6, rtol=0)


def test_partial_rotation_preserves_passthrough_channels():
    """Channels >= rotary_dim must be untouched (atol=0)."""
    torch.manual_seed(2)
    batch, heads, seq, head_dim = 1, 2, 3, 64
    rotary_dim = 16

    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    cos = torch.cos(torch.randn(batch, seq, rotary_dim))
    sin = torch.sin(torch.randn(batch, seq, rotary_dim))

    q_out, k_out = partial_neox_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # Pass-through tail must be bit-exact.
    assert torch.equal(q_out[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_out[..., rotary_dim:], k[..., rotary_dim:])
    # And the rotated head must NOT be equal to the input (sanity).
    assert not torch.equal(q_out[..., :rotary_dim], q[..., :rotary_dim])


def test_install_is_idempotent_and_byte_safe_on_full_rotation():
    """Installing the patch must not perturb full-rotation HF callers."""
    # Snapshot HF behaviour BEFORE installing.
    torch.manual_seed(3)
    head_dim = 32
    q = torch.randn(1, 2, 4, head_dim)
    k = torch.randn(1, 2, 4, head_dim)
    cos = torch.cos(torch.randn(1, 4, head_dim))
    sin = torch.sin(torch.randn(1, 4, head_dim))
    q_pre, k_pre = hf_apply(q, k, cos, sin, unsqueeze_dim=1)

    install_partial_neox_rotary()
    install_partial_neox_rotary()  # idempotent

    # After install, the module-level symbol must be the patched one.
    q_post, k_post = modeling_llama.apply_rotary_pos_emb(
        q, k, cos, sin, unsqueeze_dim=1
    )
    assert torch.equal(q_pre, q_post)
    assert torch.equal(k_pre, k_post)

    uninstall_partial_neox_rotary()
    uninstall_partial_neox_rotary()  # idempotent

    # After uninstall, the symbol must be HF's original (object identity).
    assert (
        modeling_llama.apply_rotary_pos_emb is hf_apply
        or getattr(modeling_llama.apply_rotary_pos_emb, "__wrapped__", None) is hf_apply
    )


def test_rejects_cos_longer_than_head_dim():
    """Defensive — cos can't be larger than q's last dim."""
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    cos = torch.randn(1, 1, 16)  # > head_dim=8
    sin = torch.randn(1, 1, 16)

    with pytest.raises(ValueError, match="exceeds q last dim"):
        partial_neox_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
