"""Unit tests for GLM-5 Dense MLA DFlash / DSpark draft layers."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.glm5 import (
    GLM5_DSPARK_ARCHITECTURE,
    Glm5Config,
    Glm5DFlashMLAAttention,
    apply_interleaved_rotary_pos_emb,
    mla_kwargs_from_verifier,
)
from speculators.models.dflash.model_definitions import (
    _rotate_half,
    apply_rotary_pos_emb,
)
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.train.cli import (
    _apply_glm5_lora_rank_overrides,
    create_transformer_layer_config,
)


TINY_GLM5_KWARGS = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "q_lora_rank": 16,
    "kv_lora_rank": 8,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 4,
    "v_head_dim": 8,
    "qk_head_dim": 12,
    "rope_interleave": True,
    "layer_types": ["full_attention", "full_attention"],
    "_attn_implementation": "eager",
}


def _tiny_glm5_config(**overrides) -> Glm5Config:
    kwargs = dict(TINY_GLM5_KWARGS)
    kwargs.update(overrides)
    return Glm5Config(**kwargs)


def test_autoconfig_registers_glm5():
    cfg = AutoConfig.for_model("glm5_dspark")
    assert cfg.__class__ is Glm5Config
    assert cfg.model_type == "glm5_dspark"


def test_glm5_config_defaults_lock_576d_page_spec():
    cfg = Glm5Config()
    assert cfg.model_type == "glm5_dspark"
    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_rope_head_dim == 64
    assert cfg.kv_lora_rank + cfg.qk_rope_head_dim == 576
    assert cfg.rope_interleave is True


def test_mla_kwargs_from_verifier_copies_ranks():
    verifier = SimpleNamespace(
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=192,
        qk_head_dim=256,
    )
    kwargs = mla_kwargs_from_verifier(verifier)
    assert kwargs["q_lora_rank"] == 2048
    assert kwargs["kv_lora_rank"] == 512
    assert kwargs["qk_nope_head_dim"] == 192
    assert kwargs["qk_rope_head_dim"] == 64
    assert kwargs["v_head_dim"] == 192
    assert kwargs["qk_head_dim"] == 256
    assert kwargs["rope_interleave"] is True


def test_mla_kwargs_defaults_v_head_to_qk_head_dim():
    # GLM-5.2: v_head_dim == qk_head_dim (256), not qk_nope (192).
    verifier = SimpleNamespace(
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        qk_head_dim=256,
    )
    kwargs = mla_kwargs_from_verifier(verifier)
    assert kwargs["v_head_dim"] == 256
    assert kwargs["qk_head_dim"] == 256


def test_mla_kwargs_recovers_qk_rope_when_autoconfig_clobbers_it():
    # transformers 5.5.x GlmMoe sets qk_rope_head_dim = head_dim = qk_nope.
    verifier = SimpleNamespace(
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=192,
        qk_head_dim=256,
        v_head_dim=256,
    )
    kwargs = mla_kwargs_from_verifier(verifier)
    assert kwargs["qk_nope_head_dim"] == 192
    assert kwargs["qk_rope_head_dim"] == 64
    assert kwargs["qk_head_dim"] == 256


def test_mla_kwargs_defaults_v_and_qk_head_from_nope_plus_rope():
    verifier = SimpleNamespace(
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
    )
    kwargs = mla_kwargs_from_verifier(verifier)
    assert kwargs["qk_head_dim"] == 12
    assert kwargs["v_head_dim"] == 12


def test_glm5_config_defaults_v_head_to_qk_head_dim():
    cfg = _tiny_glm5_config(v_head_dim=None, qk_head_dim=None)
    assert cfg.qk_head_dim == 12  # 8 + 4
    assert cfg.v_head_dim == 12


def test_mla_kwargs_rejects_non_mla_verifier():
    with pytest.raises(ValueError, match="requires verifier MLA fields"):
        mla_kwargs_from_verifier(SimpleNamespace(hidden_size=32))


def test_create_transformer_layer_config_copies_glm5_mla_fields():
    verifier = SimpleNamespace(
        vocab_size=128,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        head_dim=8,
        intermediate_size=64,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_head_dim=12,
    )
    with patch(
        "speculators.train.cli.AutoConfig.from_pretrained", return_value=verifier
    ):
        config = create_transformer_layer_config(
            "dummy",
            num_layers=2,
            draft_arch="glm5",
            hidden_act=None,
            sliding_window=2048,
            full_attention_indices=[0, 1],
        )
    assert isinstance(config, Glm5Config)
    assert config.model_type == "glm5_dspark"
    assert config.q_lora_rank == 16
    assert config.kv_lora_rank == 8
    assert config.qk_nope_head_dim == 8
    assert config.qk_rope_head_dim == 4
    assert config.num_attention_heads == 2
    assert config.layer_types == ["full_attention", "full_attention"]
    assert config.rope_interleave is True


def test_apply_glm5_lora_rank_overrides():
    config = _tiny_glm5_config()
    args = SimpleNamespace(q_lora_rank=4096, kv_lora_rank=2048)
    _apply_glm5_lora_rank_overrides(config, args)
    assert config.q_lora_rank == 4096
    assert config.kv_lora_rank == 2048

    config = _tiny_glm5_config()
    _apply_glm5_lora_rank_overrides(
        config, SimpleNamespace(q_lora_rank=0, kv_lora_rank=None)
    )
    assert config.q_lora_rank is None
    assert config.kv_lora_rank == 8

    with pytest.raises(ValueError, match="must be > 0"):
        _apply_glm5_lora_rank_overrides(
            _tiny_glm5_config(), SimpleNamespace(q_lora_rank=None, kv_lora_rank=0)
        )

    with pytest.raises(ValueError, match="must be >= 0"):
        _apply_glm5_lora_rank_overrides(
            _tiny_glm5_config(), SimpleNamespace(q_lora_rank=-1, kv_lora_rank=None)
        )

    with pytest.raises(ValueError, match="require --draft-arch glm5"):
        _apply_glm5_lora_rank_overrides(
            SimpleNamespace(model_type="qwen3"),
            SimpleNamespace(q_lora_rank=4096, kv_lora_rank=2048),
        )


def test_mla_interleaved_rope_differs_from_neox():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 1, 5, 4)
    # HF Qwen3RotaryEmbedding layout: cat(freqs, freqs) along the last dim.
    freqs = torch.randn(1, 5, 2)
    cos = torch.cat([freqs, freqs], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    interleaved_q, interleaved_k = apply_interleaved_rotary_pos_emb(q, k, cos, sin)
    neox_q, neox_k = apply_rotary_pos_emb(q, k, cos, sin)
    assert interleaved_q.shape == q.shape
    assert interleaved_k.shape == k.shape
    assert not torch.allclose(interleaved_q, neox_q)
    assert not torch.allclose(interleaved_k, neox_k)
    # NeoX still splits the last dim in half (control that the helper is unchanged).
    rotated = _rotate_half(q)
    half = q.shape[-1] // 2
    assert torch.equal(rotated[..., :half], -q[..., half:])
    assert torch.equal(rotated[..., half:], q[..., :half])


def test_mla_interleaved_rope_rotates_adjacent_pairs():
    # d=2: unique freq c,s; pair (x0, x1) -> (x0*c - x1*s, x1*c + x0*s).
    q = torch.tensor([[[[2.0, 3.0]]]])  # [B, H, S_q=1, D=2]
    k = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]])  # [B, 1, S=2, D=2]
    cos = torch.tensor([[[0.6, 0.6]]]).expand(1, 2, 2).contiguous()
    sin = torch.tensor([[[0.8, 0.8]]]).expand(1, 2, 2).contiguous()
    q_out, k_out = apply_interleaved_rotary_pos_emb(q, k, cos, sin)
    c, s = 0.6, 0.8
    expected_q = torch.tensor([[[[2.0 * c - 3.0 * s, 3.0 * c + 2.0 * s]]]])
    expected_k0 = torch.tensor([4.0 * c - 5.0 * s, 5.0 * c + 4.0 * s])
    # Q uses only the last q_len positions of the KV-length cache.
    assert torch.allclose(q_out, expected_q, atol=1e-6)
    assert torch.allclose(k_out[0, 0, 1], expected_k0, atol=1e-6)


def test_mla_attention_calls_interleaved_rope():
    cfg = _tiny_glm5_config()
    attn = Glm5DFlashMLAAttention(cfg, layer_idx=0)
    assert attn.rope_interleave is True
    bsz, q_len, ctx_len = 1, 2, 4
    hidden = torch.randn(bsz, q_len, cfg.hidden_size)
    target = torch.randn(bsz, ctx_len, cfg.hidden_size)
    kv_len = ctx_len + q_len
    cos = torch.randn(bsz, kv_len, cfg.qk_rope_head_dim)
    sin = torch.randn(bsz, kv_len, cfg.qk_rope_head_dim)
    with patch(
        "speculators.models.dflash.glm5.apply_interleaved_rotary_pos_emb",
        wraps=apply_interleaved_rotary_pos_emb,
    ) as mocked:
        out, _ = attn(hidden, target, (cos, sin), attention_mask=None)
    mocked.assert_called_once()
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_mla_attention_honors_rope_interleave_false():
    cfg = _tiny_glm5_config(rope_interleave=False)
    attn = Glm5DFlashMLAAttention(cfg, layer_idx=0)
    assert attn.rope_interleave is False
    bsz, q_len, ctx_len = 1, 2, 4
    hidden = torch.randn(bsz, q_len, cfg.hidden_size)
    target = torch.randn(bsz, ctx_len, cfg.hidden_size)
    kv_len = ctx_len + q_len
    cos = torch.randn(bsz, kv_len, cfg.qk_rope_head_dim)
    sin = torch.randn(bsz, kv_len, cfg.qk_rope_head_dim)
    with (
        patch(
            "speculators.models.dflash.glm5.apply_interleaved_rotary_pos_emb",
            wraps=apply_interleaved_rotary_pos_emb,
        ) as interleaved,
        patch(
            "speculators.models.dflash.glm5.apply_rotary_pos_emb",
            wraps=apply_rotary_pos_emb,
        ) as neox,
    ):
        out, _ = attn(hidden, target, (cos, sin), attention_mask=None)
    interleaved.assert_not_called()
    neox.assert_called_once()
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert torch.isfinite(out).all()


def _make_dflash_glm5() -> DFlashDraftModel:
    tl_config = _tiny_glm5_config()
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=False,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


def test_glm5_dflash_uses_mla_projections():
    model = _make_dflash_glm5()
    attn = model.layers[0].self_attn
    assert isinstance(attn, Glm5DFlashMLAAttention)
    assert hasattr(attn, "q_a_proj")
    assert hasattr(attn, "kv_a_proj_with_mqa")
    assert hasattr(attn, "kv_b_proj")
    assert not hasattr(attn, "q_proj")
    assert not hasattr(attn, "k_proj")
    assert hasattr(model, "context_proj")
    assert hasattr(model, "context_norm")
    assert not hasattr(model, "fc")
    assert not hasattr(model, "hidden_norm")
    keys = model.state_dict()
    assert "context_proj.weight" in keys
    assert "fc.weight" not in keys
    assert "layers.0.self_attn.q_a_proj.weight" in keys
    assert "layers.0.self_attn.kv_a_proj_with_mqa.weight" in keys
    assert "layers.0.self_attn.fused_qkv_a_proj.weight" not in keys
    # Rotary cache is built at qk_rope_head_dim, not the MLA v/nope dim.
    assert model.rotary_emb.config.head_dim == 4
    assert model.config.transformer_layer_config.head_dim == 8


def test_glm5_dflash_forward_finite_loss():
    torch.manual_seed(0)
    model = _make_dflash_glm5()
    seq_len = 16
    hidden_size = 32
    hidden_states = torch.randn(1, seq_len, 2 * hidden_size)
    verifier_last = torch.randn(1, seq_len, hidden_size)
    input_ids = torch.randint(0, 64, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        _, loss, metrics = model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last,
            document_ids=document_ids,
            max_anchors=2,
        )
    assert torch.isfinite(loss)
    assert metrics is not None


def test_glm5_dspark_config_roundtrip():
    tl_config = _tiny_glm5_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
    )
    dumped = config.model_dump()
    reloaded = DSparkSpeculatorConfig(**dumped)
    assert reloaded.transformer_layer_config.model_type == "glm5_dspark"
    assert reloaded.transformer_layer_config.q_lora_rank == 16
    assert reloaded.transformer_layer_config.kv_lora_rank == 8
    assert reloaded.architectures == [GLM5_DSPARK_ARCHITECTURE]


def test_glm5_dspark_builds_mla_layers():
    tl_config = _tiny_glm5_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
    )
    model = DSparkDraftModel(config)
    assert isinstance(model.layers[0].self_attn, Glm5DFlashMLAAttention)
    assert model.markov_head is not None
    assert hasattr(model, "context_proj")
    assert config.architectures == [GLM5_DSPARK_ARCHITECTURE]


def test_glm5_dspark_save_pretrained_export_contract(tmp_path):
    tl_config = _tiny_glm5_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        enable_confidence_head=False,
    )
    model = DSparkDraftModel(config)
    out = tmp_path / "glm5_dspark"
    model.save_pretrained(out)
    saved = json.loads((out / "config.json").read_text())
    assert saved["architectures"] == [GLM5_DSPARK_ARCHITECTURE]
    assert saved["transformer_layer_config"]["model_type"] == "glm5_dspark"
    assert saved["transformer_layer_config"]["kv_lora_rank"] == 8
    assert saved["transformer_layer_config"]["q_lora_rank"] == 16
    # Reloading a nested dict that still says legacy "glm5" upgrades the type.
    nested = dict(saved["transformer_layer_config"])
    nested["model_type"] = "glm5"
    upgraded = DSparkSpeculatorConfig(
        transformer_layer_config=nested,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        enable_confidence_head=False,
    )
    assert upgraded.transformer_layer_config.model_type == "glm5_dspark"
    assert upgraded.architectures == [GLM5_DSPARK_ARCHITECTURE]


def test_glm5_dspark_push_to_hub_uploads_after_architecture_rewrite(tmp_path):
    tl_config = _tiny_glm5_config()
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        enable_confidence_head=False,
    )
    model = DSparkDraftModel(config)
    out = tmp_path / "glm5_dspark"
    uploaded: dict[str, object] = {}

    def _upload(save_directory, repo_id, _files_timestamps, **_kwargs):
        uploaded["architectures"] = json.loads(
            (Path(save_directory) / "config.json").read_text()
        )["architectures"]
        uploaded["repo_id"] = repo_id
        return "ok"

    with (
        patch(
            "huggingface_hub.create_repo",
            return_value=SimpleNamespace(repo_id="user/glm5-dspark"),
        ),
        patch.object(DSparkDraftModel, "_get_files_timestamps", return_value={}),
        patch.object(DSparkDraftModel, "_upload_modified_files", side_effect=_upload),
    ):
        model.save_pretrained(out, push_to_hub=True, repo_id="user/glm5-dspark")

    assert uploaded["architectures"] == [GLM5_DSPARK_ARCHITECTURE]
    assert uploaded["repo_id"] == "user/glm5-dspark"
    saved = json.loads((out / "config.json").read_text())
    assert saved["architectures"] == [GLM5_DSPARK_ARCHITECTURE]
