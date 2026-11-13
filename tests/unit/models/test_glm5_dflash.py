"""Unit tests for GLM-5 Dense MLA DFlash / DSpark draft layers."""

import argparse
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    apply_rotary_pos_emb_interleave,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.config import (
    GLM5_DSPARK_MODEL_TYPE,
    Glm5DSparkDecoderConfig,
)
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.model_definitions import (
    DFlashMLAAttention,
    DFlashMLADecoderLayer,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    _apply_mla_rope,
    apply_rotary_pos_emb,
    dflash_model_classes,
)
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.utils import mla_kwargs_from_verifier
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
    "rope_interleave": True,
    "layer_types": ["full_attention", "full_attention"],
    "_attn_implementation": "eager",
}


def _tiny_glm5_config(**overrides) -> Glm5DSparkDecoderConfig:
    kwargs = dict(TINY_GLM5_KWARGS)
    kwargs.update(overrides)
    return Glm5DSparkDecoderConfig(**kwargs)


def test_autoconfig_registers_glm5():
    cfg = AutoConfig.for_model("glm5_dspark")
    assert cfg.__class__ is Glm5DSparkDecoderConfig
    assert isinstance(cfg, DeepseekV3Config)
    assert cfg.model_type == "glm5_dspark"


def test_dflash_model_classes_table():
    glm5 = dflash_model_classes[GLM5_DSPARK_MODEL_TYPE]
    assert glm5.decoder_layer_class is DFlashMLADecoderLayer
    assert issubclass(glm5.decoder_layer_class.attention_class, DeepseekV3Attention)
    assert glm5.rotary_head_dim_attr == "qk_rope_head_dim"
    qwen3 = dflash_model_classes["qwen3"]
    assert qwen3.decoder_layer_class is Qwen3DFlashDecoderLayer
    assert qwen3.rotary_head_dim_attr is None


def test_qwen3_dflash_keeps_fc_and_hidden_norm():
    tl_config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    model = DFlashDraftModel(
        DFlashSpeculatorConfig(
            transformer_layer_config=tl_config,
            draft_vocab_size=64,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
        )
    )
    assert isinstance(model.layers[0].self_attn, Qwen3DFlashAttention)
    assert hasattr(model, "fc")
    assert hasattr(model, "hidden_norm")
    assert model.rotary_emb.config.head_dim == 16
    keys = model.state_dict()
    assert "fc.weight" in keys
    assert "hidden_norm.weight" in keys


def test_glm5_config_defaults_lock_576d_page_spec():
    # Upstream DeepSeek-V3 defaults already give the serveable 576-d MLA page.
    cfg = Glm5DSparkDecoderConfig()
    assert cfg.model_type == "glm5_dspark"
    assert cfg.kv_lora_rank + cfg.qk_rope_head_dim == 576
    assert cfg.rope_interleave is True
    assert cfg.qk_head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim


def test_glm5_config_defaults_are_dense_only():
    cfg = _tiny_glm5_config(num_hidden_layers=3)
    assert cfg.n_routed_experts == 0
    assert cfg.n_shared_experts == 0
    assert cfg.num_experts_per_tok == 0
    assert cfg.num_mtp_layers == 0
    # No layer index reaches the MoE range.
    assert cfg.first_k_dense_replace == cfg.num_hidden_layers == 3
    exported = cfg.to_diff_dict()
    assert exported["n_routed_experts"] == 0
    assert exported["first_k_dense_replace"] == 3
    # Explicit values still win over the dense defaults.
    assert _tiny_glm5_config(first_k_dense_replace=1).first_k_dense_replace == 1
    # Round-trip through the serialized dict keeps the dense semantics.
    reloaded = Glm5DSparkDecoderConfig(**cfg.to_diff_dict())
    assert reloaded.first_k_dense_replace == 3
    assert reloaded.n_routed_experts == 0


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
    assert "qk_head_dim" not in kwargs  # derived by DeepseekV3Config
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


def test_mla_kwargs_defaults_v_head_from_nope_plus_rope():
    verifier = SimpleNamespace(
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
    )
    kwargs = mla_kwargs_from_verifier(verifier)
    assert kwargs["v_head_dim"] == 12


def test_glm5_config_derives_qk_head_dim():
    cfg = _tiny_glm5_config()
    assert cfg.qk_head_dim == 12  # 8 + 4


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
    assert isinstance(config, Glm5DSparkDecoderConfig)
    assert config.model_type == "glm5_dspark"
    assert config.q_lora_rank == 16
    assert config.kv_lora_rank == 8
    assert config.qk_nope_head_dim == 8
    assert config.qk_rope_head_dim == 4
    assert config.qk_head_dim == 12
    assert config.num_attention_heads == 2
    assert config.num_key_value_heads == 2
    assert config.layer_types == ["full_attention", "full_attention"]
    assert config.rope_interleave is True


def test_apply_glm5_lora_rank_overrides():
    config = _tiny_glm5_config()
    args = argparse.Namespace(q_lora_rank=4096, kv_lora_rank=2048)
    _apply_glm5_lora_rank_overrides(config, args)
    assert config.q_lora_rank == 4096
    assert config.kv_lora_rank == 2048

    config = _tiny_glm5_config()
    _apply_glm5_lora_rank_overrides(
        config, argparse.Namespace(q_lora_rank=0, kv_lora_rank=None)
    )
    assert config.q_lora_rank is None
    assert config.kv_lora_rank == 8

    with pytest.raises(ValueError, match="must be > 0"):
        _apply_glm5_lora_rank_overrides(
            _tiny_glm5_config(), argparse.Namespace(q_lora_rank=None, kv_lora_rank=0)
        )

    with pytest.raises(ValueError, match="must be >= 0"):
        _apply_glm5_lora_rank_overrides(
            _tiny_glm5_config(), argparse.Namespace(q_lora_rank=-1, kv_lora_rank=None)
        )

    with pytest.raises(ValueError, match="require --draft-arch glm5"):
        _apply_glm5_lora_rank_overrides(
            Qwen3Config(),
            argparse.Namespace(q_lora_rank=4096, kv_lora_rank=2048),
        )


def _rope_cache(kv_len: int, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    # HF rotary layout: cat(freqs, freqs) along the last dim.
    angles = torch.randn(1, kv_len, rope_dim // 2)
    return (
        torch.cat([angles.cos(), angles.cos()], dim=-1),
        torch.cat([angles.sin(), angles.sin()], dim=-1),
    )


def test_apply_mla_rope_rotates_q_at_trailing_positions():
    torch.manual_seed(0)
    kv_len, q_len, rope_dim = 5, 3, 4
    k = torch.randn(1, 1, kv_len, rope_dim)
    q = k[..., -q_len:, :].clone()  # same vectors as the last q_len K positions
    cos, sin = _rope_cache(kv_len, rope_dim)
    for interleave in (True, False):
        q_out, k_out = _apply_mla_rope(interleave, q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        # Q must be rotated with the trailing positions of the KV-length cache.
        assert torch.allclose(q_out, k_out[..., -q_len:, :], atol=1e-6)
    # Interleaved (pairwise) and NeoX (split-half) rotation genuinely differ,
    # and the interleaved path is the upstream DeepSeek helper.
    inter_q, _ = _apply_mla_rope(True, q, k, cos, sin)
    neox_q, _ = _apply_mla_rope(False, q, k, cos, sin)
    assert not torch.allclose(inter_q, neox_q)
    ref_q, _ = apply_rotary_pos_emb_interleave(
        q, q, cos[..., -q_len:, :], sin[..., -q_len:, :]
    )
    assert torch.allclose(inter_q, ref_q)


def _run_tiny_mla_attention(cfg: Glm5DSparkDecoderConfig) -> torch.Tensor:
    attn = DFlashMLAAttention(cfg, layer_idx=0)
    bsz, q_len, ctx_len = 1, 2, 4
    hidden = torch.randn(bsz, q_len, cfg.hidden_size)
    target = torch.randn(bsz, ctx_len, cfg.hidden_size)
    cos, sin = _rope_cache(ctx_len + q_len, cfg.qk_rope_head_dim)
    out, _ = attn(hidden, target, (cos, sin), attention_mask=None)
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert torch.isfinite(out).all()
    return out


def test_mla_attention_reuses_upstream_and_calls_interleaved_rope():
    cfg = _tiny_glm5_config()
    attn = DFlashMLAAttention(cfg, layer_idx=0)
    assert isinstance(attn, DeepseekV3Attention)
    assert attn.is_causal is False
    assert attn.num_key_value_groups == 1
    assert attn.scaling == pytest.approx(cfg.qk_head_dim**-0.5)
    with patch(
        "speculators.models.dflash.model_definitions.apply_rotary_pos_emb_interleave",
        wraps=apply_rotary_pos_emb_interleave,
    ) as mocked:
        _run_tiny_mla_attention(cfg)
    assert mocked.call_count == 2  # once for Q, once for K


def test_mla_attention_honors_rope_interleave_false():
    cfg = _tiny_glm5_config(rope_interleave=False)
    with (
        patch(
            "speculators.models.dflash.model_definitions.apply_rotary_pos_emb_interleave",
            wraps=apply_rotary_pos_emb_interleave,
        ) as interleaved,
        patch(
            "speculators.models.dflash.model_definitions.apply_rotary_pos_emb",
            wraps=apply_rotary_pos_emb,
        ) as neox,
    ):
        _run_tiny_mla_attention(cfg)
    interleaved.assert_not_called()
    neox.assert_called_once()


def test_mla_attention_applies_yarn_mscale_from_upstream():
    cfg = _tiny_glm5_config(
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 32.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 64,
        }
    )
    attn = DFlashMLAAttention(cfg, layer_idx=0)
    mscale = 0.1 * 1.0 * math.log(32.0) + 1.0
    assert attn.scaling == pytest.approx(cfg.qk_head_dim**-0.5 * mscale * mscale)


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
    assert isinstance(attn, DFlashMLAAttention)
    assert attn.q_a_proj is not None
    assert attn.kv_a_proj_with_mqa is not None
    assert attn.kv_b_proj is not None
    # Upstream keeps a ``q_proj = None`` placeholder when Q LoRA is enabled.
    assert getattr(attn, "q_proj", None) is None
    assert not hasattr(attn, "k_proj")
    assert hasattr(model, "fc")
    assert hasattr(model, "hidden_norm")
    keys = model.state_dict()
    assert "fc.weight" in keys
    assert "hidden_norm.weight" in keys
    assert "context_proj.weight" not in keys
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
    assert isinstance(reloaded.transformer_layer_config, Glm5DSparkDecoderConfig)
    assert reloaded.transformer_layer_config.model_type == "glm5_dspark"
    assert reloaded.transformer_layer_config.q_lora_rank == 16
    assert reloaded.transformer_layer_config.kv_lora_rank == 8


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
    assert isinstance(model.layers[0].self_attn, DFlashMLAAttention)
    assert model.markov_head is not None
    assert hasattr(model, "fc")


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
    assert saved["architectures"] == ["DSparkDraftModel"]
    assert saved["transformer_layer_config"]["model_type"] == "glm5_dspark"
    assert saved["transformer_layer_config"]["kv_lora_rank"] == 8
    assert saved["transformer_layer_config"]["q_lora_rank"] == 16
    weights = out / "model.safetensors"
    assert weights.exists()
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(weights), framework="pt") as f:
        keys = set(f.keys())
    assert "fc.weight" in keys
    assert "hidden_norm.weight" in keys
    assert "context_proj.weight" not in keys
    reloaded = DSparkSpeculatorConfig.from_pretrained(out)
    assert isinstance(reloaded.transformer_layer_config, Glm5DSparkDecoderConfig)
    assert reloaded.to_dict()["architectures"] == ["DSparkDraftModel"]


def test_qwen3_dspark_keeps_python_architecture_name(tmp_path):
    tl_config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=8,
        enable_confidence_head=False,
    )
    out = tmp_path / "qwen3_dspark"
    DSparkDraftModel(config).save_pretrained(out)
    saved = json.loads((out / "config.json").read_text())
    assert saved["architectures"] == ["DSparkDraftModel"]
