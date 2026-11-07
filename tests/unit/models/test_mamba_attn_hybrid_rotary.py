"""Regression tests for the H-Spec (mamba_attn_hybrid) partial RoPE repair.

The drafter shares the target's *already-rotated* K prefix, so its Q/K
rotation must reproduce the target's effective text/full-attention rope:
``rotary_dim = head_dim * partial_rotary_factor`` inverse frequencies,
neox pairing ``j <-> j + rotary_dim/2`` inside the rotated prefix, and an
unchanged channel suffix. The stock ``Qwen3RotaryEmbedding`` computes
frequencies from the full ``head_dim`` and ignores ``partial_rotary_factor``
— the bug these tests pin against.

All rotation references here are hand-coded independently of the
implementation under test (mirroring vLLM's ``RotaryEmbedding`` /
``MRotaryEmbedding`` text path); no test rotates both sides with the same
code.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import PreTrainedModel, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

import speculators.models.mamba_attn_hybrid.model_definitions as hspec_model_definitions
from speculators.models.eagle3.rotary_partial import (
    partial_neox_apply_rotary_pos_emb,
)
from speculators.models.mamba_attn_hybrid.model_definitions import (
    VerifierKVAttention,
)
from speculators.models.mamba_attn_hybrid.rotary import (
    MambaAttnHybridRotaryEmbedding,
    effective_rope,
    reconcile_rope_with_verifier,
    resolve_target_rope,
    validate_checkpoint_rope_compat,
)

HEAD_DIM = 256
ROTARY_DIM = 64
THETA = 10_000_000.0
PARTIAL = 0.25

# Qwen3.8-27B-style target rope (MRoPE metadata included: it must be
# normalized away — for scalar text positions an MRotaryEmbedding applies
# the same ordinary partial neox rotation).
TARGET_ROPE = {
    "rope_type": "default",
    "rope_theta": THETA,
    "partial_rotary_factor": PARTIAL,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


def _verifier_dir(tmp_path, rope_parameters=TARGET_ROPE, head_dim=HEAD_DIM):
    cfg = Qwen3Config(
        hidden_size=5120,
        num_hidden_layers=8,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=head_dim,
        vocab_size=248320,
        rope_parameters=dict(rope_parameters),
    )
    d = tmp_path / "verifier"
    d.mkdir(exist_ok=True)
    cfg.save_pretrained(d)
    return str(d)


def _draft_config_dict():
    return {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": HEAD_DIM,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 262144,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }


# ---------------------------------------------------------------------------
# Independent references (written from the vLLM spec, not from the code under
# test).
# ---------------------------------------------------------------------------


def _ref_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )


def _ref_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float):
    """Width-``rotary_dim`` neox cos/sin tables, float64 reference."""
    inv = _ref_inv_freq(rotary_dim, theta)
    freqs = torch.outer(positions.double(), inv)  # (seq, rotary_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _ref_partial_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate the first ``cos.shape[-1]`` channels; pass the suffix through.

    ``x``: (..., head_dim); ``cos``/``sin``: (seq, rotary_dim).
    """
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    out_rot = x_rot * cos + rotated * sin
    return torch.cat((out_rot, x_pass), dim=-1)


# ---------------------------------------------------------------------------
# 1. Resolved config
# ---------------------------------------------------------------------------


def test_resolve_target_rope_normalizes_mrope(tmp_path):
    verifier = _verifier_dir(tmp_path)
    head_dim, rope = resolve_target_rope(verifier)
    assert head_dim == HEAD_DIM
    assert rope == {
        "rope_type": "default",
        "rope_theta": THETA,
        "partial_rotary_factor": PARTIAL,
    }


def test_reconcile_explicit_draft_config_without_rope(tmp_path):
    verifier = _verifier_dir(tmp_path)
    draft_json = tmp_path / "draft_config.json"
    draft_json.write_text(json.dumps(_draft_config_dict()))
    cfg = Qwen3Config.from_dict(_draft_config_dict())
    # Before: Qwen3 defaults — full head, theta 1e4 (the bug).
    assert effective_rope(cfg) == (HEAD_DIM, HEAD_DIM, 10000.0)
    reconcile_rope_with_verifier(cfg, verifier, draft_config_path=str(draft_json))
    assert effective_rope(cfg) == (HEAD_DIM, ROTARY_DIM, THETA)
    # Saved/reloaded config retains it.
    serialized = {k: v for k, v in cfg.to_diff_dict().items() if "rope" in k}
    assert serialized["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": THETA,
        "partial_rotary_factor": PARTIAL,
    }
    reloaded = Qwen3Config.from_dict(cfg.to_diff_dict())
    assert effective_rope(reloaded) == (HEAD_DIM, ROTARY_DIM, THETA)


def test_reconcile_automatic_path(tmp_path):
    """No --draft-config: the automatic decoder-config path is covered too."""
    verifier = _verifier_dir(tmp_path)
    cfg = Qwen3Config.from_dict(_draft_config_dict())
    reconcile_rope_with_verifier(cfg, verifier, draft_config_path=None)
    assert effective_rope(cfg) == (HEAD_DIM, ROTARY_DIM, THETA)


def test_reconcile_explicit_compatible_override(tmp_path):
    verifier = _verifier_dir(tmp_path)
    d = _draft_config_dict()
    d["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": THETA,
        "partial_rotary_factor": PARTIAL,
    }
    draft_json = tmp_path / "draft_config.json"
    draft_json.write_text(json.dumps(d))
    cfg = Qwen3Config.from_dict(d)
    reconcile_rope_with_verifier(cfg, verifier, draft_config_path=str(draft_json))
    assert effective_rope(cfg) == (HEAD_DIM, ROTARY_DIM, THETA)


def test_reconcile_explicit_incompatible_override_rejected(tmp_path):
    verifier = _verifier_dir(tmp_path)
    d = _draft_config_dict()
    d["rope_parameters"] = {"rope_type": "default", "rope_theta": 500000.0}
    draft_json = tmp_path / "draft_config.json"
    draft_json.write_text(json.dumps(d))
    cfg = Qwen3Config.from_dict(d)
    with pytest.raises(ValueError, match="incompatible"):
        reconcile_rope_with_verifier(cfg, verifier, draft_config_path=str(draft_json))


def test_reconcile_rope_dim_override_rejected(tmp_path):
    """A rope_dim override incompatible with the shared target K is refused."""
    verifier = _verifier_dir(tmp_path)
    d = _draft_config_dict()
    d["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": THETA,
        "rope_dim": 128,
    }
    draft_json = tmp_path / "draft_config.json"
    draft_json.write_text(json.dumps(d))
    cfg = Qwen3Config.from_dict(d)
    with pytest.raises(ValueError, match="rotary_dim 128 != target 64"):
        reconcile_rope_with_verifier(cfg, verifier, draft_config_path=str(draft_json))


def test_reconcile_head_dim_mismatch_rejected(tmp_path):
    verifier = _verifier_dir(tmp_path)
    d = _draft_config_dict()
    d["head_dim"] = 128
    cfg = Qwen3Config.from_dict(d)
    with pytest.raises(ValueError, match="head_dim"):
        reconcile_rope_with_verifier(cfg, verifier)


def test_checkpoint_compat_diagnostic(tmp_path):
    verifier = _verifier_dir(tmp_path)
    good = {
        "transformer_layer_config": {
            **_draft_config_dict(),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": THETA,
                "partial_rotary_factor": PARTIAL,
            },
        }
    }
    good_dir = tmp_path / "good_ckpt"
    good_dir.mkdir()
    (good_dir / "config.json").write_text(json.dumps(good))
    validate_checkpoint_rope_compat(str(good_dir), verifier)  # no raise

    # Old wrong-rope checkpoint (rope omitted -> full-head theta 1e4).
    bad_dir = tmp_path / "bad_ckpt"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text(
        json.dumps({"transformer_layer_config": _draft_config_dict()})
    )
    with pytest.raises(ValueError, match="incompatible.*rotary_dim"):
        validate_checkpoint_rope_compat(str(bad_dir), verifier)


# ---------------------------------------------------------------------------
# 2. Actual tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 7, 127, 2048, 8191])
def test_tables_match_formula(position):
    cfg = Qwen3Config.from_dict(
        {
            **_draft_config_dict(),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": THETA,
                "partial_rotary_factor": PARTIAL,
            },
        }
    )
    rot = MambaAttnHybridRotaryEmbedding(cfg)
    assert rot.inv_freq.numel() == ROTARY_DIM // 2
    assert torch.allclose(
        rot.inv_freq,
        _ref_inv_freq(ROTARY_DIM, THETA).float(),
        atol=1e-7,
        rtol=1e-6,
    )
    x = torch.zeros(1, 1, 8, dtype=torch.float32)
    cos, sin = rot(x, torch.tensor([[position]]))
    assert cos.shape[-1] == ROTARY_DIM
    assert sin.shape[-1] == ROTARY_DIM
    ref_cos, ref_sin = _ref_cos_sin(torch.tensor([position]), ROTARY_DIM, THETA)
    # float32 phase computation tolerance: at position 8191 the phase reaches
    # ~8191 rad, where fp32 ulp (~5e-4) propagates into sin/cos.
    torch.testing.assert_close(cos[0, 0], ref_cos[0].float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(sin[0, 0], ref_sin[0].float(), atol=1e-3, rtol=1e-3)


def test_tables_are_duplicated_halves():
    """neox layout: channel j and j+rotary_dim/2 share a frequency."""
    cfg = Qwen3Config.from_dict(
        {
            **_draft_config_dict(),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": THETA,
                "partial_rotary_factor": PARTIAL,
            },
        }
    )
    rot = MambaAttnHybridRotaryEmbedding(cfg)
    x = torch.zeros(1, 1, 8, dtype=torch.float32)
    cos, sin = rot(x, torch.tensor([[33]]))
    torch.testing.assert_close(cos[0, 0, :32], cos[0, 0, 32:])
    torch.testing.assert_close(sin[0, 0, :32], sin[0, 0, 32:])


# ---------------------------------------------------------------------------
# 3. Pairing and suffix via the real attention path inputs
# ---------------------------------------------------------------------------


def _partial_cfg():
    return Qwen3Config.from_dict(
        {
            **_draft_config_dict(),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": THETA,
                "partial_rotary_factor": PARTIAL,
            },
        }
    )


def test_rotation_pairs_0_with_32_not_128():
    """Basis-vector test: channel 0 rotates against 32, never 128."""
    rot = MambaAttnHybridRotaryEmbedding(_partial_cfg())
    pos = torch.tensor([[5]])
    cos, sin = rot(torch.zeros(1, 1, 8), pos)
    # q = e_0 (channel 0 one-hot); rotation moves mass only to channel 32.
    q = torch.zeros(1, 1, 1, HEAD_DIM)
    q[..., 0] = 1.0
    q_rot, _ = partial_neox_apply_rotary_pos_emb(q, q.clone(), cos, sin)
    q_rot = q_rot[0, 0, 0]
    assert q_rot[32] == pytest.approx(torch.sin(5 * rot.inv_freq[0]).item(), abs=2e-5)
    assert q_rot[0] == pytest.approx(torch.cos(5 * rot.inv_freq[0]).item(), abs=2e-5)
    assert q_rot[128].abs() < 1e-7  # 0 <-> 128 would be the full-head pairing
    assert q_rot[64:].abs().sum() < 1e-7  # suffix untouched


def test_q_k_match_independent_reference_and_suffix_bitwise():
    torch.manual_seed(0)
    rot = MambaAttnHybridRotaryEmbedding(_partial_cfg())
    seq = 6
    pos = torch.arange(seq).unsqueeze(0)
    cos, sin = rot(torch.zeros(1, seq, 8), pos)
    q = torch.randn(1, 4, seq, HEAD_DIM)
    k = torch.randn(1, 2, seq, HEAD_DIM)
    q_rot, k_rot = partial_neox_apply_rotary_pos_emb(q, k, cos, sin)
    ref_cos, ref_sin = _ref_cos_sin(pos[0], ROTARY_DIM, THETA)
    for got, src in ((q_rot, q), (k_rot, k)):
        ref = torch.stack(
            [
                _ref_partial_neox(src[0, h, t], ref_cos[t].float(), ref_sin[t].float())
                for h in range(src.shape[1])
                for t in range(seq)
            ]
        ).view_as(src)
        torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
        # Channels 64:256 bitwise unchanged.
        assert torch.equal(got[..., ROTARY_DIM:], src[..., ROTARY_DIM:])


# ---------------------------------------------------------------------------
# 4. Target-prefix interaction through the real VerifierKVAttention.forward
# ---------------------------------------------------------------------------


def test_verifier_kv_attention_preserves_target_prefix(monkeypatch, tmp_path):
    cfg = _partial_cfg()
    cfg.layer_types = ["full_attention"] * 5
    cfg.sliding_window = None
    attn_layer = VerifierKVAttention(cfg, layer_idx=0)
    attn_layer.eval()
    assert attn_layer.scaling == pytest.approx(1.0 / 16.0)

    torch.manual_seed(1)
    num_kv_heads = cfg.num_key_value_heads
    prefix_len, q_len = 3, 2
    ctx_pos = torch.tensor([10, 11, 12])
    draft_pos = torch.tensor([13, 14])

    # Independently target-rotated context K (reference implementation).
    k_ctx = torch.randn(1, num_kv_heads, prefix_len, HEAD_DIM)
    ref_cos_c, ref_sin_c = _ref_cos_sin(ctx_pos, ROTARY_DIM, THETA)
    k_ctx_rot = torch.stack(
        [
            _ref_partial_neox(
                k_ctx[0, h, t], ref_cos_c[t].float(), ref_sin_c[t].float()
            )
            for h in range(num_kv_heads)
            for t in range(prefix_len)
        ]
    ).view_as(k_ctx)
    v_ctx = torch.randn(1, num_kv_heads, prefix_len, HEAD_DIM)

    hidden = torch.randn(1, q_len, cfg.hidden_size)
    rot = MambaAttnHybridRotaryEmbedding(cfg)
    cos, sin = rot(torch.zeros(1, q_len, 8), draft_pos.unsqueeze(0))

    captured = {}

    def fake_flex_attention(self, q, k, v, attention_mask, scaling, **kw):
        captured.update(q=q, k=k, v=v, scaling=scaling)
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[3]), None

    monkeypatch.setattr(
        hspec_model_definitions, "flex_attention_forward", fake_flex_attention
    )
    with torch.no_grad():
        attn_layer(
            hidden,
            (k_ctx_rot, v_ctx),
            (cos, sin),
            attention_mask=None,
        )

    q, k, v = captured["q"], captured["k"], captured["v"]
    assert captured["scaling"] == pytest.approx(1.0 / 16.0)
    # Target prefix K appended without another rotation (bitwise).
    assert torch.equal(k[:, :, :prefix_len], k_ctx_rot)
    # Target prefix V preserved.
    assert torch.equal(v[:, :, :prefix_len], v_ctx)
    # New Q/K match the independent partial-neox reference at draft positions.
    q_proj = attn_layer.q_norm(
        attn_layer.q_proj(hidden).view(1, q_len, -1, HEAD_DIM)
    ).transpose(1, 2)
    k_proj = attn_layer.k_norm(
        attn_layer.k_proj(hidden).view(1, q_len, -1, HEAD_DIM)
    ).transpose(1, 2)
    v_proj = attn_layer.v_proj(hidden).view(1, q_len, -1, HEAD_DIM).transpose(1, 2)
    ref_cos_d, ref_sin_d = _ref_cos_sin(draft_pos, ROTARY_DIM, THETA)
    for got, src in ((q, q_proj), (k[:, :, prefix_len:], k_proj)):
        ref = torch.stack(
            [
                _ref_partial_neox(
                    src[0, h, t], ref_cos_d[t].float(), ref_sin_d[t].float()
                )
                for h in range(src.shape[1])
                for t in range(q_len)
            ]
        ).view_as(src)
        torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
    # Draft V is the unrotated projection.
    assert torch.equal(v[:, :, prefix_len:], v_proj)
    # Prefix attention logits use scale 1/16 against the rotated prefix K
    # (query head 0 attends its KV group head 0 under GQA).
    logits = (q[0, 0, :1] @ k_ctx_rot[0, 0].T) * (1.0 / 16.0)
    ref_q = _ref_partial_neox(
        q_proj[0, 0, 0], ref_cos_d[0].float(), ref_sin_d[0].float()
    )
    ref_logits = (ref_q @ k_ctx_rot[0, 0].T) * (1.0 / 16.0)
    torch.testing.assert_close(logits[0], ref_logits, atol=2e-4, rtol=2e-4)


# ---------------------------------------------------------------------------
# 5. Full-head regression (partial_rotary_factor == 1.0)
# ---------------------------------------------------------------------------


def test_full_head_factor_one_matches_stock_qwen3():
    full = Qwen3Config(
        hidden_size=5120,
        num_hidden_layers=2,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=100,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    rot = MambaAttnHybridRotaryEmbedding(full)
    stock = Qwen3RotaryEmbedding(full)
    assert rot.inv_freq.numel() == 64
    pos = torch.tensor([[0, 3, 777]])
    x = torch.zeros(1, 3, 8, dtype=torch.float32)
    cos_a, sin_a = rot(x, pos)
    cos_b, sin_b = stock(x, pos)
    assert cos_a.shape == cos_b.shape
    torch.testing.assert_close(cos_a, cos_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sin_a, sin_b, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 6. Checkpoint round trip: training rotation == serving reconstruction
# ---------------------------------------------------------------------------


def _serving_get_rope_reconstruction(rope_parameters: dict, head_dim: int):
    """Mimic vLLM ``get_rope``'s default path (no mrope_section, no rope_dim):
    rotary_dim = int(head_size * partial_rotary_factor), inv_freq over
    rotary_dim with rope_theta base."""
    base = rope_parameters.get("rope_theta", 10000)
    partial = rope_parameters.get("partial_rotary_factor", 1.0)
    rotary_dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )
    return rotary_dim, inv_freq


def test_checkpoint_round_trip_training_matches_serving(tmp_path):
    cfg = _partial_cfg()
    rot = MambaAttnHybridRotaryEmbedding(cfg)
    # Serialize as the saved speculator config would.
    saved = cfg.to_diff_dict()
    reloaded = Qwen3Config.from_dict(saved)
    rot2 = MambaAttnHybridRotaryEmbedding(reloaded)
    torch.testing.assert_close(rot.inv_freq, rot2.inv_freq, atol=1e-6, rtol=1e-6)

    serving_rotary_dim, serving_inv_freq = _serving_get_rope_reconstruction(
        reloaded.rope_parameters, HEAD_DIM
    )
    assert serving_rotary_dim == ROTARY_DIM
    # inv_freq is float32 in training; the reference is float64.
    torch.testing.assert_close(
        rot.inv_freq.double(), serving_inv_freq, atol=1e-7, rtol=1e-7
    )


def test_hf_init_weights_preserves_partial_frequencies():
    """HF ``PreTrainedModel._init_weights`` re-initializes any module named
    ``*RotaryEmbedding*`` holding ``original_inv_freq`` via
    ``compute_default_rope_parameters`` — it must reproduce the partial
    frequencies, not the stock full-head ones (this crashed/corrupted
    training before the hook existed)."""

    class Holder(PreTrainedModel):
        config_class = Qwen3Config

        def __init__(self, config):
            super().__init__(config)
            self.rotary_emb = MambaAttnHybridRotaryEmbedding(config)
            self.post_init()

    holder = Holder(_partial_cfg())
    expected = 1.0 / (THETA ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    torch.testing.assert_close(
        holder.rotary_emb.inv_freq, expected, atol=1e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        holder.rotary_emb.original_inv_freq, expected, atol=1e-7, rtol=1e-6
    )
