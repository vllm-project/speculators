"""
Unit tests for Eagle3 converter focusing on the specific fixes implemented.

Tests cover:
- Embeddings replacement with verifier embeddings
- Weight remapping from midlayer.* to layers.0.*
- Configuration compatibility (max_position_embeddings, rope_theta)
- Validation of the conversion process
"""

from unittest.mock import patch

import pytest
import torch

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.eagle.utils import (
    load_checkpoint_config,
)


class TestEagle3ConverterFixes:
    """Test the specific fixes implemented in Eagle3 converter."""

    @pytest.fixture
    def sample_eagle3_config(self):
        """Sample Eagle3 configuration for testing."""
        return {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 2048,  # This will be maxed with verifier
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 10000.0,
            "mlp_bias": False,
        }

    @pytest.fixture
    def sample_eagle3_weights(self):
        """Sample Eagle3 weights with midlayer.* naming."""
        return {
            # Embeddings that should be replaced
            "embed_tokens.weight": torch.randn(128000, 4096),
            # Weights that should be remapped from midlayer.* to layers.0.*
            "midlayer.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "midlayer.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "midlayer.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "midlayer.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "midlayer.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "midlayer.mlp.up_proj.weight": torch.randn(11008, 4096),
            "midlayer.mlp.down_proj.weight": torch.randn(4096, 11008),
            "midlayer.input_layernorm.weight": torch.randn(4096),
            "midlayer.post_attention_layernorm.weight": torch.randn(4096),
            # Other weights
            "t2d": torch.randn(128000, 4096),
        }

    @pytest.fixture
    def sample_verifier_config(self):
        """Sample verifier configuration."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 131072,  # Should be maxed with Eagle3's 2048
            "rope_theta": 500000.0,
            "vocab_size": 128256,
        }

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_max_position_embeddings_logic(
        self, mock_get_config, sample_eagle3_config, sample_verifier_config
    ):
        """Test that max_position_embeddings uses max of Eagle3 and verifier."""
        mock_get_config.return_value = (sample_verifier_config, None)

        converter = Eagle3Converter()

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )

        # Check that max_position_embeddings is the max of both values
        # Eagle3: 2048, Verifier: 131072, so should be 131072
        assert llama_config.max_position_embeddings == 131072

        # Check that other values come from Eagle3 config
        assert llama_config.hidden_size == 4096
        assert llama_config.num_attention_heads == 32
        # rope_theta comes from Eagle3 config, not verifier
        assert llama_config.rope_theta == 10000.0

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_num_hidden_layers_from_config(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test that num_hidden_layers is taken from eagle_config when present."""
        mock_get_config.return_value = ({}, None)
        converter = Eagle3Converter()

        # Add num_hidden_layers to the sample config
        sample_eagle3_config["num_hidden_layers"] = 3

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama_config.num_hidden_layers == 3

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_num_hidden_layers_default(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test that num_hidden_layers defaults to 1 when not in config."""
        mock_get_config.return_value = ({}, None)
        converter = Eagle3Converter()

        # Remove num_hidden_layers if present
        sample_eagle3_config.pop("num_hidden_layers", None)

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama_config.num_hidden_layers == 1

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_fallback_when_verifier_unavailable(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test fallback behavior when verifier config cannot be loaded."""
        mock_get_config.side_effect = Exception("Network error")

        converter = Eagle3Converter()

        # Should raise exception since method doesn't handle this gracefully
        with pytest.raises((RuntimeError, ValueError)):
            converter._create_transformer_config_from_eagle(
                sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
            )

    @pytest.mark.sanity
    def test_converted_model_config_structure(self):
        """Test that the config structure created is valid for Eagle3Speculator."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            config = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
            )

        # Verify config structure
        assert config.speculators_model_type == "eagle3"
        assert config.transformer_layer_config.hidden_size == 4096
        assert config.speculators_config.algorithm == "eagle3"
        assert (
            config.speculators_config.verifier.name_or_path
            == "meta-llama/Llama-3.1-8B-Instruct"
        )

    @pytest.mark.sanity
    def test_eagle_aux_hidden_state_layer_ids_parameter(self):
        """Test that eagle_aux_hidden_state_layer_ids parameter is properly handled."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            # Test with None (default)
            config_none = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
            )
            assert config_none.eagle_aux_hidden_state_layer_ids is None

            # Test with specific layer IDs
            layer_ids = [1, 23, 44]
            config_with_ids = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
                eagle_aux_hidden_state_layer_ids=layer_ids,
            )
            assert config_with_ids.eagle_aux_hidden_state_layer_ids == layer_ids

    @pytest.mark.sanity
    def test_eagle_aux_hidden_state_layer_ids_in_config_serialization(self):
        """Test that eagle_aux_hidden_state_layer_ids is properly serialized."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            layer_ids = [1, 23, 44]
            config = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
                eagle_aux_hidden_state_layer_ids=layer_ids,
            )

            # Test that the config can be serialized to dict
            config_dict = config.to_dict()
            assert "eagle_aux_hidden_state_layer_ids" in config_dict
            assert config_dict["eagle_aux_hidden_state_layer_ids"] == layer_ids

    @pytest.mark.sanity
    def test_nm_testing_2layer_eagle3_model_config(self, tmp_path):
        """Test that multi-layer conversion support."""
        converter = Eagle3Converter()

        # TODO: Use a real model in the future  # noqa: FIX002
        checkpoint_path = "nm-testing/testing-llama3.1.8b-2layer-eagle3"
        base_model = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"

        converter.convert(
            checkpoint_path,
            tmp_path / checkpoint_path.rsplit("/", maxsplit=1)[-1],
            base_model,
            norm_before_residual=False,
        )

        config = load_checkpoint_config(
            tmp_path / checkpoint_path.rsplit("/", maxsplit=1)[-1]
        )

        # Verify that num_hidden_layers is correctly set to 2
        assert config["transformer_layer_config"]["num_hidden_layers"] == 2
