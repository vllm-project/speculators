"""Unit tests for DFlashConverter config building and weight remapping."""

import logging
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.dflash.converter import DFlashConverter
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

_HIDDEN = 16
_NUM_HEADS = 2
_NUM_KV_HEADS = 1
_HEAD_DIM = 8
_Q_DIM = _NUM_HEADS * _HEAD_DIM  # 16
_KV_DIM = _NUM_KV_HEADS * _HEAD_DIM  # 8


def _tiny_dflash_config(num_aux_layers=1):
    return DFlashSpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            vocab_size=32,
            hidden_size=_HIDDEN,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=_NUM_HEADS,
            num_key_value_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            max_position_embeddings=32,
        ),
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=list(range(num_aux_layers)),
        mask_token_id=1,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
        ),
    )


def _source_config(**overrides):
    config = {
        "model_type": "qwen3",
        "architectures": ["DFlashDraftModel"],
        "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
        "block_size": 16,
        "num_target_layers": 36,
        "dflash_config": {
            "mask_token_id": 151669,
            "target_layer_ids": [1, 9, 17, 25, 33],
        },
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    config.update(overrides)
    return config


class TestBuildConfig:
    @patch("speculators.convert.dflash.converter.PretrainedConfig.get_config_dict")
    def test_happy_path(self, mock_get_config):
        mock_get_config.return_value = (
            {
                "hidden_size": 4096,
                "num_hidden_layers": 36,
                "architectures": ["Qwen3ForCausalLM"],
            },
            None,
        )
        config = DFlashConverter()._build_config(
            _source_config(), "Qwen/Qwen3-8B", None
        )

        assert config.speculators_config.algorithm == "dflash"
        assert config.speculators_config.verifier.name_or_path == "Qwen/Qwen3-8B"
        assert config.block_size == 16
        assert config.draft_vocab_size == 151936
        assert config.mask_token_id == 151669
        assert config.speculators_config.proposal_methods[0].speculative_tokens == 15  # type: ignore[attr-defined]
        # z-lab target_layer_ids are offset by +1 to speculators layer ids
        assert config.aux_hidden_state_layer_ids == [2, 10, 18, 26, 34]
        # non-transformer keys are stripped from transformer_layer_config
        assert config.transformer_layer_config.num_hidden_layers == 5
        assert not hasattr(config.transformer_layer_config, "dflash_config")

    @patch("speculators.convert.dflash.converter.PretrainedConfig.get_config_dict")
    def test_excludes_last_verifier_layer(self, mock_get_config):
        mock_get_config.return_value = (
            {
                "hidden_size": 4096,
                "num_hidden_layers": 34,
                "architectures": ["Qwen3ForCausalLM"],
            },
            None,
        )
        # target_layer_ids [1, 9, 17, 25, 33] → +1 → [2, 10, 18, 26, 34]
        # but 34 == num_hidden_layers, so it should be excluded
        config = DFlashConverter()._build_config(
            _source_config(), "Qwen/Qwen3-8B", None
        )
        assert config.aux_hidden_state_layer_ids == [2, 10, 18, 26]

    @patch("speculators.convert.dflash.converter.PretrainedConfig.get_config_dict")
    def test_explicit_aux_layer_ids_override(self, mock_get_config):
        mock_get_config.return_value = ({"hidden_size": 4096}, None)
        config = DFlashConverter()._build_config(
            _source_config(), "Qwen/Qwen3-8B", [3, 11, 19]
        )
        assert config.aux_hidden_state_layer_ids == [3, 11, 19]

    @patch("speculators.convert.dflash.converter.PretrainedConfig.get_config_dict")
    def test_hidden_size_mismatch_raises(self, mock_get_config):
        mock_get_config.return_value = ({"hidden_size": 2048}, None)
        with pytest.raises(ValueError, match="Architecture mismatch"):
            DFlashConverter()._build_config(_source_config(), "some/model", None)

    @patch("speculators.convert.dflash.converter.PretrainedConfig.get_config_dict")
    def test_missing_target_layer_ids_raises(self, mock_get_config):
        mock_get_config.return_value = (
            {"hidden_size": 4096, "num_hidden_layers": 36},
            None,
        )
        source = _source_config(dflash_config={"mask_token_id": 151669})
        with pytest.raises(ValueError, match="target_layer_ids"):
            DFlashConverter()._build_config(source, "Qwen/Qwen3-8B", None)


class TestRemapWeights:
    def _make_fused_weights(self):
        qkv = torch.randn(_Q_DIM + 2 * _KV_DIM, _HIDDEN)
        return {
            "layers.0.self_attn.qkv_proj.weight": qkv,
            "layers.0.self_attn.g_proj.weight": torch.randn(2, _HIDDEN),
            "layers.0.self_attn.o_proj.weight": torch.randn(_HIDDEN, _Q_DIM),
            "aux_hidden_norms.0.weight": torch.randn(_HIDDEN),
            "fc.weight": torch.randn(_HIDDEN, _HIDDEN * 2),
            "norm.weight": torch.randn(_HIDDEN),
        }

    def test_splits_fused_qkv(self):
        config = _tiny_dflash_config()
        model = DFlashDraftModel(config=config)
        weights = self._make_fused_weights()
        qkv = weights["layers.0.self_attn.qkv_proj.weight"]

        remapped = DFlashConverter()._remap_weights(weights, config, model)

        assert "layers.0.self_attn.qkv_proj.weight" not in remapped
        assert torch.equal(remapped["layers.0.self_attn.q_proj.weight"], qkv[:_Q_DIM])
        assert torch.equal(
            remapped["layers.0.self_attn.k_proj.weight"],
            qkv[_Q_DIM : _Q_DIM + _KV_DIM],
        )
        assert torch.equal(
            remapped["layers.0.self_attn.v_proj.weight"],
            qkv[_Q_DIM + _KV_DIM :],
        )

    def test_drops_g_proj_and_aux_hidden_norms(self):
        config = _tiny_dflash_config()
        model = DFlashDraftModel(config=config)
        remapped = DFlashConverter()._remap_weights(
            self._make_fused_weights(), config, model
        )
        assert not any("g_proj" in k for k in remapped)
        assert not any("aux_hidden_norms" in k for k in remapped)

    def test_slices_fc_weight(self):
        config = _tiny_dflash_config(num_aux_layers=1)
        model = DFlashDraftModel(config=config)
        weights = self._make_fused_weights()
        # fc from checkpoint is wider than model expects
        wide_fc = torch.randn(_HIDDEN, _HIDDEN * 3)
        weights["fc.weight"] = wide_fc

        remapped = DFlashConverter()._remap_weights(weights, config, model)
        assert remapped["fc.weight"].shape[1] == model.fc.in_features
        assert torch.equal(remapped["fc.weight"], wide_fc[:, : model.fc.in_features])

    def test_passthrough_when_no_fused_qkv(self):
        config = _tiny_dflash_config()
        model = DFlashDraftModel(config=config)
        weights = {"norm.weight": torch.randn(_HIDDEN)}
        remapped = DFlashConverter()._remap_weights(weights, config, model)
        assert remapped is weights

    def test_preserves_other_keys(self):
        config = _tiny_dflash_config()
        model = DFlashDraftModel(config=config)
        weights = self._make_fused_weights()
        remapped = DFlashConverter()._remap_weights(weights, config, model)
        assert "layers.0.self_attn.o_proj.weight" in remapped
        assert "norm.weight" in remapped

    def test_mixed_fused_and_separate_qkv_raises(self):
        config = _tiny_dflash_config()
        model = DFlashDraftModel(config=config)
        weights = self._make_fused_weights()
        weights["layers.0.self_attn.q_proj.weight"] = torch.randn(_Q_DIM, _HIDDEN)
        with pytest.raises(ValueError, match="both fused qkv_proj and separate"):
            DFlashConverter()._remap_weights(weights, config, model)


class TestRopeParameters:
    _ROPE_SLIDING = {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }

    def test_nested_rope_params_flattened_with_full_attn(self):
        config = _tiny_dflash_config()
        tl = config.transformer_layer_config
        tl.layer_types = ["full_attention"]
        tl.rope_parameters = {
            "sliding_attention": dict(self._ROPE_SLIDING),
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        }
        model = DFlashDraftModel(config=config)
        assert model.rotary_emb is not None

    def test_nested_rope_params_no_warn_all_sliding(self, caplog):
        config = _tiny_dflash_config()
        tl = config.transformer_layer_config
        tl.layer_types = ["sliding_attention"]
        tl.sliding_window = 512
        tl.rope_parameters = {
            "sliding_attention": dict(self._ROPE_SLIDING),
        }
        with caplog.at_level(logging.WARNING, logger="speculators.models.dflash.core"):
            DFlashDraftModel(config=config)
        assert not any("full-attention layer" in r.message for r in caplog.records)


class TestSave:
    def test_missing_draft_weights_raise(self, tmp_path):
        # No source weights: every draft-body weight (fc, norm, hidden_norm,
        # layers.*) is missing and must be flagged, not silently kept as NaN.
        # Raises before load_verifier_weights, so no verifier download.
        with pytest.raises(ValueError, match="Draft weights missing"):
            DFlashConverter()._save(_tiny_dflash_config(), {}, tmp_path)
