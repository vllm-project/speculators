"""Unit tests for DFlashConverter config building."""

from unittest.mock import patch

import pytest
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.dflash.converter import DFlashConverter
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _tiny_dflash_config():
    return DFlashSpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
        ),
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0],
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
            {"hidden_size": 4096, "architectures": ["Qwen3ForCausalLM"]},
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
        assert config.speculators_config.proposal_methods[0].speculative_tokens == 15
        # z-lab target_layer_ids are offset by +1 to speculators layer ids
        assert config.aux_hidden_state_layer_ids == [2, 10, 18, 26, 34]
        # non-transformer keys are stripped from transformer_layer_config
        assert config.transformer_layer_config.num_hidden_layers == 5
        assert not hasattr(config.transformer_layer_config, "dflash_config")

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
        mock_get_config.return_value = ({"hidden_size": 4096}, None)
        source = _source_config(dflash_config={"mask_token_id": 151669})
        with pytest.raises(ValueError, match="target_layer_ids"):
            DFlashConverter()._build_config(source, "Qwen/Qwen3-8B", None)


class TestSave:
    def test_missing_draft_weights_raise(self, tmp_path):
        # No source weights: every draft-body weight (fc, norm, hidden_norm,
        # layers.*) is missing and must be flagged, not silently kept as NaN.
        # Raises before load_verifier_weights, so no verifier download.
        with pytest.raises(ValueError, match="Draft weights missing"):
            DFlashConverter()._save(_tiny_dflash_config(), {}, tmp_path)
