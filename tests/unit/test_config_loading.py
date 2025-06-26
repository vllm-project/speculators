"""
Unit tests for SpeculatorModelConfig loading and saving functionality.

Tests the polymorphic config loading system including:
- from_pretrained with registry-based instantiation
- from_dict with proper type resolution
- to_dict with correct serialization
- Round-trip save/load preservation
"""

import json
import tempfile
from pathlib import Path

import pytest
from transformers import LlamaConfig

from speculators.config import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.eagle import EagleSpeculatorConfig


class TestConfigLoading:
    """Test config loading and saving functionality."""

    @staticmethod
    def get_test_speculators_config():
        """Create a minimal valid SpeculatorsConfig for testing."""
        return SpeculatorsConfig(
            algorithm="eagle",
            proposal_methods=[],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="meta-llama/Llama-3.1-8B",
                architectures=["LlamaForCausalLM"],
                tokenizer="meta-llama/Llama-3.1-8B",
                vocab_size=128256,
                hidden_size=4096,
                intermediate_size=14336,
                max_position_embeddings=131072,
                bos_token_id=128000,
                eos_token_id=[128001, 128008, 128009],
            ),
        )

    def test_eagle_config_to_dict(self):
        """
        Test that EagleSpeculatorConfig.to_dict properly serializes
        transformer_layer_config.
        """
        # Create config with LlamaConfig
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=1,
        )
        eagle_config = EagleSpeculatorConfig(
            transformer_layer_config=llama_config,
            layernorms=True,
            fusion_bias=False,
            speculators_config=self.get_test_speculators_config(),
        )

        # Convert to dict
        config_dict = eagle_config.to_dict()

        # Check that transformer_layer_config is a dict
        assert isinstance(config_dict["transformer_layer_config"], dict)
        assert config_dict["transformer_layer_config"]["vocab_size"] == 32000
        assert config_dict["transformer_layer_config"]["hidden_size"] == 4096
        assert config_dict["speculators_model_type"] == "eagle"
        assert config_dict["layernorms"] is True
        assert config_dict["fusion_bias"] is False

    def test_eagle_config_from_dict(self):
        """
        Test that EagleSpeculatorConfig.from_dict properly reconstructs
        transformer_layer_config.
        """
        # Create dict with transformer_layer_config as dict
        config_dict = {
            "speculators_model_type": "eagle",
            "layernorms": True,
            "fusion_bias": True,
            "transformer_layer_config": {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 1,
            },
        }

        # Create config from dict
        eagle_config = EagleSpeculatorConfig.from_dict(config_dict)

        # Check that transformer_layer_config is a LlamaConfig instance
        assert isinstance(eagle_config.transformer_layer_config, LlamaConfig)
        assert eagle_config.transformer_layer_config.vocab_size == 32000
        assert eagle_config.transformer_layer_config.hidden_size == 4096
        assert eagle_config.layernorms is True
        assert eagle_config.fusion_bias is True

    def test_base_config_from_dict_with_registry(self):
        """
        Test that SpeculatorModelConfig.from_dict uses registry for polymorphic
        instantiation.
        """
        config_dict = {
            "speculators_model_type": "eagle",
            "layernorms": False,
            "fusion_bias": False,
            "transformer_layer_config": {
                "model_type": "llama",
                "vocab_size": 128256,
                "hidden_size": 4096,
                "num_hidden_layers": 1,
            },
        }

        # Load using base class - should return EagleSpeculatorConfig
        config = SpeculatorModelConfig.from_dict(config_dict)

        assert isinstance(config, EagleSpeculatorConfig)
        assert config.speculators_model_type == "eagle"
        assert config.layernorms is False
        assert config.fusion_bias is False

    def test_base_config_from_dict_missing_model_type(self):
        """Test that from_dict raises error when speculators_model_type is missing."""
        config_dict = {
            "layernorms": False,
            "fusion_bias": False,
        }

        with pytest.raises(
            ValueError, match="speculators_model_type must be specified"
        ):
            SpeculatorModelConfig.from_dict(config_dict)

    def test_base_config_from_dict_unknown_model_type(self):
        """Test that from_dict raises error for unknown model types."""
        config_dict = {
            "speculators_model_type": "unknown_type",
            "layernorms": False,
        }

        with pytest.raises(
            ValueError, match="Unknown speculators_model_type: unknown_type"
        ):
            SpeculatorModelConfig.from_dict(config_dict)

    def test_round_trip_save_load(self):
        """Test that configs can be saved and loaded correctly."""
        # Create original config
        original_config = EagleSpeculatorConfig(
            layernorms=True,
            fusion_bias=True,
            transformer_layer_config=LlamaConfig(
                vocab_size=32000,
                hidden_size=2048,
                num_hidden_layers=1,
            ),
            speculators_config=self.get_test_speculators_config(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "config"
            save_path.mkdir()

            # Save config
            original_config.save_pretrained(save_path)

            # Check that config.json was created
            config_file = save_path / "config.json"
            assert config_file.exists()

            # Load config back using base class
            loaded_config = SpeculatorModelConfig.from_pretrained(save_path)

            # Should be an EagleSpeculatorConfig instance
            assert isinstance(loaded_config, EagleSpeculatorConfig)
            assert loaded_config.speculators_model_type == "eagle"
            assert loaded_config.layernorms is True
            assert loaded_config.fusion_bias is True
            assert loaded_config.transformer_layer_config.vocab_size == 32000
            assert loaded_config.transformer_layer_config.hidden_size == 2048

    def test_from_pretrained_infers_eagle_from_architectures(self):
        """Test that from_pretrained can infer eagle type from architectures field."""
        config_dict = {
            "architectures": ["EagleSpeculator"],
            "layernorms": False,
            "fusion_bias": False,
            "transformer_layer_config": {
                "model_type": "llama",
                "vocab_size": 128256,
                "hidden_size": 4096,
                "num_hidden_layers": 1,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with config_path.open("w") as f:
                json.dump(config_dict, f)

            # Load without speculators_model_type - should infer from architectures
            config = SpeculatorModelConfig.from_pretrained(tmpdir)

            assert isinstance(config, EagleSpeculatorConfig)
            assert config.speculators_model_type == "eagle"

    def test_from_pretrained_no_inference_possible(self):
        """Test that from_pretrained raises error when type cannot be determined."""
        config_dict = {
            "some_field": "some_value",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with config_path.open("w") as f:
                json.dump(config_dict, f)

            with pytest.raises(
                ValueError, match="Could not determine speculators_model_type"
            ):
                SpeculatorModelConfig.from_pretrained(tmpdir)

    def test_extra_kwargs_override(self):
        """Test that kwargs override values from config dict."""
        config_dict = {
            "speculators_model_type": "eagle",
            "layernorms": False,
            "fusion_bias": False,
            "transformer_layer_config": {
                "model_type": "llama",
                "vocab_size": 128256,
                "hidden_size": 4096,
                "num_hidden_layers": 1,
            },
        }

        # Override layernorms and fusion_bias with kwargs
        config = SpeculatorModelConfig.from_dict(
            config_dict, layernorms=True, fusion_bias=True
        )

        assert isinstance(config, EagleSpeculatorConfig)
        assert config.layernorms is True  # Overridden
        assert config.fusion_bias is True  # Overridden

    def test_extra_pretrained_config_attributes(self):
        """Test that extra PretrainedConfig attributes are preserved."""
        config_dict = {
            "speculators_model_type": "eagle",
            "layernorms": False,
            "fusion_bias": False,
            "transformer_layer_config": {
                "model_type": "llama",
                "vocab_size": 128256,
                "hidden_size": 4096,
                "num_hidden_layers": 1,
            },
            # Extra PretrainedConfig attributes
            "_name_or_path": "test-model",
            "transformers_version": "4.36.0",
            "custom_attribute": "custom_value",
        }

        config = SpeculatorModelConfig.from_dict(config_dict)

        # Check that extra attributes are preserved
        assert hasattr(config, "_name_or_path")
        assert config._name_or_path == "test-model"
        assert hasattr(config, "transformers_version")
        assert config.transformers_version == "4.36.0"
        assert hasattr(config, "custom_attribute")
        assert config.custom_attribute == "custom_value"
