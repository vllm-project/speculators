"""Unit tests for LlamaEagleSpeculatorConfig."""

import pytest

from speculators.models.llama_eagle_config import (
    LlamaDecoderParameters,
    LlamaEagleSpeculatorConfig,
)


class TestLlamaDecoderParameters:
    """Test suite for LlamaDecoderParameters."""

    def test_default_values(self):
        """Test default parameter values."""
        params = LlamaDecoderParameters()
        
        # Test Llama 3.1 8B defaults
        assert params.vocab_size == 128256
        assert params.hidden_size == 4096
        assert params.intermediate_size == 14336
        assert params.num_attention_heads == 32
        assert params.num_key_value_heads == 8
        assert params.hidden_act == "silu"
        assert params.max_position_embeddings == 131072
        assert params.rms_norm_eps == 1e-5
        assert params.rope_theta == 500000.0
        assert params.attention_bias is False
        assert params.attention_dropout == 0.0
        assert params.mlp_bias is False
        assert params.pad_token_id is None
        assert params.bos_token_id == 128000
        assert params.eos_token_id == 128001

    def test_custom_values(self):
        """Test setting custom parameter values."""
        params = LlamaDecoderParameters(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=5632,
            num_attention_heads=16,
            num_key_value_heads=16,
            rope_theta=10000.0,
        )
        
        assert params.vocab_size == 32000
        assert params.hidden_size == 2048
        assert params.intermediate_size == 5632
        assert params.num_attention_heads == 16
        assert params.num_key_value_heads == 16
        assert params.rope_theta == 10000.0


class TestLlamaEagleSpeculatorConfig:
    """Test suite for LlamaEagleSpeculatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LlamaEagleSpeculatorConfig()
        
        # Test model type
        assert config.speculators_model_type == "llama_eagle"
        assert config.architectures == ["LlamaEagleSpeculator"]
        
        # Test EAGLE v1 defaults
        assert config.fc_bias is False
        assert config.use_extra_layernorms is False
        assert config.extra_layernorm_positions is None
        
        # Test transformer defaults
        assert config.inputs == ["input_ids", "hidden_states[-1]"]
        assert config.replace_first_layer_norm is True
        assert config.num_hidden_layers == 1
        
        # Test decoder params
        assert isinstance(config.llama_decoder_params, LlamaDecoderParameters)

    def test_eagle_v1_config(self):
        """Test EAGLE v1 configuration."""
        config = LlamaEagleSpeculatorConfig(
            fc_bias=False,
            use_extra_layernorms=False,
            num_hidden_layers=2,
        )
        
        assert config.fc_bias is False
        assert config.use_extra_layernorms is False
        assert config.num_hidden_layers == 2

    def test_hass_config(self):
        """Test HASS configuration."""
        config = LlamaEagleSpeculatorConfig(
            fc_bias=True,
            use_extra_layernorms=True,
            extra_layernorm_positions=["post_embedding", "pre_lm_head"],
            num_hidden_layers=3,
        )
        
        assert config.fc_bias is True
        assert config.use_extra_layernorms is True
        assert config.extra_layernorm_positions == ["post_embedding", "pre_lm_head"]
        assert config.num_hidden_layers == 3

    def test_custom_llama_decoder_params(self):
        """Test custom LlamaDecoderParameters."""
        custom_params = LlamaDecoderParameters(
            vocab_size=32000,
            hidden_size=2048,
        )
        
        config = LlamaEagleSpeculatorConfig(
            llama_decoder_params=custom_params
        )
        
        assert config.llama_decoder_params.vocab_size == 32000
        assert config.llama_decoder_params.hidden_size == 2048

    def test_dict_conversion(self):
        """Test conversion to/from dict."""
        config = LlamaEagleSpeculatorConfig(
            fc_bias=True,
            num_hidden_layers=2,
            llama_decoder_params=LlamaDecoderParameters(
                vocab_size=50000,
                hidden_size=1024,
            )
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Check values in dict
        assert config_dict["speculators_model_type"] == "llama_eagle"
        assert config_dict["fc_bias"] is True
        assert config_dict["num_hidden_layers"] == 2
        assert config_dict["llama_decoder_params"]["vocab_size"] == 50000
        assert config_dict["llama_decoder_params"]["hidden_size"] == 1024
        
        # Convert back from dict
        loaded_config = LlamaEagleSpeculatorConfig.from_dict(config_dict)
        
        assert loaded_config.fc_bias is True
        assert loaded_config.num_hidden_layers == 2
        assert loaded_config.llama_decoder_params.vocab_size == 50000
        assert loaded_config.llama_decoder_params.hidden_size == 1024

    def test_registry_registration(self):
        """Test that config is properly registered."""
        from speculators.config import SpeculatorModelConfig
        
        # Check that llama_eagle is registered
        registry = SpeculatorModelConfig.get_registry()
        assert "llama_eagle" in registry
        assert registry["llama_eagle"] == LlamaEagleSpeculatorConfig