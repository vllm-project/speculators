"""Unit tests for MTP converter.

Tests cover:
- Key remapping (native mtp.* -> speculators mtp_layers.0.* format)
- MoE expert weight fusing (individual per-expert -> packed 3D tensors)
- MTP format verification (rejects checkpoints without mtp.* keys)
- Config building (speculators_config, verifier metadata)
- Weight extraction and model saving
"""

from unittest.mock import patch

import pytest
import torch

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.mtp.converter import (
    _EMBED_KEYS,
    _LM_HEAD_KEY,
    EXACT_KEY_MAP,
    PREFIX_KEY_MAP,
    MTPConverter,
)
from speculators.models.mtp import MTPConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


class TestKeyRemapping:
    """Test key remapping from native format to speculators format."""

    def test_embed_tokens_remapped(self):
        for embed_key in _EMBED_KEYS:
            assert MTPConverter._remap_key(embed_key) == "embed_tokens.weight"

    def test_lm_head_passthrough(self):
        assert MTPConverter._remap_key(_LM_HEAD_KEY) == _LM_HEAD_KEY

    def test_exact_key_map(self):
        for src, dst in EXACT_KEY_MAP.items():
            assert MTPConverter._remap_key(src) == dst

    def test_prefix_key_map(self):
        for src_prefix, dst_prefix in PREFIX_KEY_MAP:
            suffix = "weight"
            assert (
                MTPConverter._remap_key(f"{src_prefix}{suffix}")
                == f"{dst_prefix}{suffix}"
            )

    def test_unknown_key_passthrough(self):
        key = "model.layers.0.self_attn.q_proj.weight"
        assert MTPConverter._remap_key(key) == key

    def test_fc_weight_remapped(self):
        assert (
            MTPConverter._remap_key("mtp.fc.weight") == "mtp_layers.0.input_proj.weight"
        )

    def test_norm_weight_remapped(self):
        assert (
            MTPConverter._remap_key("mtp.norm.weight")
            == "mtp_layers.0.final_layernorm.weight"
        )

    def test_hidden_layernorm_remapped(self):
        assert (
            MTPConverter._remap_key("mtp.pre_fc_norm_hidden.weight")
            == "mtp_layers.0.hidden_layernorm.weight"
        )

    def test_token_layernorm_remapped(self):
        assert (
            MTPConverter._remap_key("mtp.pre_fc_norm_embedding.weight")
            == "mtp_layers.0.token_layernorm.weight"
        )

    def test_layer_key_remapped(self):
        assert (
            MTPConverter._remap_key("mtp.layers.0.self_attn.q_proj.weight")
            == "mtp_layers.0.self_attn.q_proj.weight"
        )


class TestMoEExpertFusing:
    """Test MoE expert weight fusing for Qwen3-Next style checkpoints."""

    @pytest.fixture
    def dense_weights(self):
        return {
            "mtp_layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
        }

    @pytest.fixture
    def moe_weights(self):
        num_experts = 4
        weights = {}
        for i in range(num_experts):
            weights[f"mtp_layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
                32, 64
            )
            weights[f"mtp_layers.0.mlp.experts.{i}.up_proj.weight"] = torch.randn(
                32, 64
            )
            weights[f"mtp_layers.0.mlp.experts.{i}.down_proj.weight"] = torch.randn(
                64, 32
            )
        weights["mtp_layers.0.self_attn.q_proj.weight"] = torch.randn(64, 64)
        return weights

    @pytest.mark.sanity
    def test_no_experts_passthrough(self, dense_weights):
        result = MTPConverter._fuse_moe_experts(dense_weights)
        assert result == dense_weights

    @pytest.mark.sanity
    def test_experts_fused_to_gate_up_proj(self, moe_weights):
        result = MTPConverter._fuse_moe_experts(moe_weights)
        assert "mtp_layers.0.mlp.experts.gate_up_proj" in result
        gate_up = result["mtp_layers.0.mlp.experts.gate_up_proj"]
        assert gate_up.shape == (4, 64, 64)

    @pytest.mark.sanity
    def test_experts_fused_to_down_proj(self, moe_weights):
        result = MTPConverter._fuse_moe_experts(moe_weights)
        assert "mtp_layers.0.mlp.experts.down_proj" in result
        down = result["mtp_layers.0.mlp.experts.down_proj"]
        assert down.shape == (4, 64, 32)

    @pytest.mark.sanity
    def test_non_expert_keys_preserved(self, moe_weights):
        result = MTPConverter._fuse_moe_experts(moe_weights)
        assert "mtp_layers.0.self_attn.q_proj.weight" in result

    @pytest.mark.sanity
    def test_individual_expert_keys_removed(self, moe_weights):
        result = MTPConverter._fuse_moe_experts(moe_weights)
        for key in moe_weights:
            if ".experts." in key and key.split(".experts.")[1][0].isdigit():
                assert key not in result

    def test_non_contiguous_expert_indices_raises(self):
        weights = {
            "mtp_layers.0.mlp.experts.0.gate_proj.weight": torch.randn(32, 64),
            "mtp_layers.0.mlp.experts.0.up_proj.weight": torch.randn(32, 64),
            "mtp_layers.0.mlp.experts.0.down_proj.weight": torch.randn(64, 32),
            "mtp_layers.0.mlp.experts.2.gate_proj.weight": torch.randn(32, 64),
            "mtp_layers.0.mlp.experts.2.up_proj.weight": torch.randn(32, 64),
            "mtp_layers.0.mlp.experts.2.down_proj.weight": torch.randn(64, 32),
        }
        with pytest.raises(ValueError, match="Non-contiguous expert indices"):
            MTPConverter._fuse_moe_experts(weights)

    def test_missing_proj_raises(self):
        weights = {
            "mtp_layers.0.mlp.experts.0.gate_proj.weight": torch.randn(32, 64),
            "mtp_layers.0.mlp.experts.0.up_proj.weight": torch.randn(32, 64),
            # missing down_proj for expert 0
        }
        with pytest.raises(ValueError, match="missing down_proj"):
            MTPConverter._fuse_moe_experts(weights)

    @pytest.mark.sanity
    def test_gate_up_content_correct(self):
        gate = torch.randn(32, 64)
        up = torch.randn(32, 64)
        weights = {
            "mtp_layers.0.mlp.experts.0.gate_proj.weight": gate,
            "mtp_layers.0.mlp.experts.0.up_proj.weight": up,
            "mtp_layers.0.mlp.experts.0.down_proj.weight": torch.randn(64, 32),
        }
        result = MTPConverter._fuse_moe_experts(weights)
        gate_up = result["mtp_layers.0.mlp.experts.gate_up_proj"]
        assert torch.equal(gate_up[0, :32], gate)
        assert torch.equal(gate_up[0, 32:], up)


class TestMTPFormatVerification:
    """Test that the converter correctly rejects non-MTP checkpoints."""

    @pytest.mark.sanity
    def test_rejects_non_mtp_checkpoint(self):
        converter = MTPConverter()
        keys = ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]
        with pytest.raises(ValueError, match="No keys with prefix 'mtp.'"):
            converter._verify_mtp_format(keys)

    @pytest.mark.sanity
    def test_accepts_mtp_checkpoint(self):
        converter = MTPConverter()
        keys = ["mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight"]
        converter._verify_mtp_format(keys)


class TestConfigBuilding:
    """Test MTPConfig construction from source config and verifier metadata."""

    @pytest.mark.sanity
    @patch("speculators.convert.mtp.converter.PretrainedConfig.get_config_dict")
    def test_config_has_correct_algorithm(self, mock_get_config):
        mock_get_config.return_value = (
            {"architectures": ["Qwen3NextForCausalLM"]},
            None,
        )
        converter = MTPConverter()
        source_config = {
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "vocab_size": 1000,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
        }
        config = converter._build_config(
            source_config, "Qwen/Qwen3-Next-80B-A3B-Instruct", 3
        )
        assert config.speculators_config.algorithm == "mtp"
        assert config.speculators_model_type == "mtp"

    @pytest.mark.sanity
    @patch("speculators.convert.mtp.converter.PretrainedConfig.get_config_dict")
    def test_config_speculative_steps(self, mock_get_config):
        mock_get_config.return_value = (
            {"architectures": ["Qwen3NextForCausalLM"]},
            None,
        )
        converter = MTPConverter()
        source_config = {
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "vocab_size": 1000,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
        }
        config = converter._build_config(
            source_config, "Qwen/Qwen3-Next-80B-A3B-Instruct", 5
        )
        assert config.num_speculative_steps == 5

    @pytest.mark.sanity
    @patch("speculators.convert.mtp.converter.PretrainedConfig.get_config_dict")
    def test_config_verifier_metadata(self, mock_get_config):
        mock_get_config.return_value = (
            {"architectures": ["Qwen3NextForCausalLM"]},
            None,
        )
        converter = MTPConverter()
        source_config = {
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "vocab_size": 1000,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
        }
        config = converter._build_config(
            source_config, "Qwen/Qwen3-Next-80B-A3B-Instruct", 3
        )
        assert (
            config.speculators_config.verifier.name_or_path
            == "Qwen/Qwen3-Next-80B-A3B-Instruct"
        )
        assert config.speculators_config.verifier.architectures == [
            "Qwen3NextForCausalLM"
        ]

    @pytest.mark.sanity
    @patch("speculators.convert.mtp.converter.PretrainedConfig.get_config_dict")
    def test_config_preserves_source_config(self, mock_get_config):
        mock_get_config.return_value = (
            {"architectures": ["Qwen3NextForCausalLM"]},
            None,
        )
        converter = MTPConverter()
        source_config = {
            "model_type": "qwen3_next",
            "hidden_size": 256,
            "vocab_size": 2000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
        }
        config = converter._build_config(
            source_config, "Qwen/Qwen3-Next-80B-A3B-Instruct", 3
        )
        assert config.transformer_layer_config.hidden_size == 256
        assert config.transformer_layer_config.vocab_size == 2000


class TestSaveAndValidate:
    """Test model saving and validation logic."""

    @pytest.mark.sanity
    def test_save_rejects_unexpected_keys(self, tmp_path):
        config = MTPConfig(
            transformer_layer_config={  # type: ignore[arg-type]
                "model_type": "qwen3",
                "hidden_size": 64,
                "vocab_size": 100,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
            },
            speculators_config=SpeculatorsConfig(
                algorithm="mtp",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(name_or_path=None, architectures=[]),
            ),
        )
        weights = {"totally_unexpected_key": torch.randn(10)}
        converter = MTPConverter()
        with pytest.raises(ValueError, match="Unexpected keys"):
            converter._save(config, weights, tmp_path / "out")
