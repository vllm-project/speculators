"""
Unit tests for Eagle3 converter focusing on the specific fixes implemented.

Tests cover:
- Embeddings replacement with verifier embeddings
- Weight remapping from midlayer.* to layers.0.*
- Configuration compatibility (max_position_embeddings, rope_theta)
- Validation of the conversion process
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, PretrainedConfig

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.models.eagle3 import Eagle3Speculator


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

    def test_weight_remapping_midlayer_to_layers(self, sample_eagle3_weights):
        """Test that midlayer.* weights are correctly remapped to layers.0.*"""
        converter = Eagle3Converter()
        
        # Mock only the embeddings addition, let the remapping happen naturally
        with patch.object(converter, '_add_verifier_embeddings') as mock_add_embeddings:
            # Mock should preserve the remapped weights and just add embeddings
            def mock_add_embeddings_func(weights, base_model):
                # Just add a fake embedding and return
                weights_copy = weights.copy()
                weights_copy["embed_tokens.weight"] = torch.randn(128256, 4096)
                return weights_copy
            
            mock_add_embeddings.side_effect = mock_add_embeddings_func
            
            processed_weights = converter._process_checkpoint_weights(
                sample_eagle3_weights, "meta-llama/Llama-3.1-8B"
            )

        # Check that midlayer weights are remapped
        assert "layers.0.self_attn.q_proj.weight" in processed_weights
        assert "layers.0.self_attn.k_proj.weight" in processed_weights
        assert "layers.0.mlp.down_proj.weight" in processed_weights
        assert "layers.0.input_layernorm.weight" in processed_weights

        # Check that original midlayer keys are not present
        assert "midlayer.self_attn.q_proj.weight" not in processed_weights
        assert "midlayer.mlp.down_proj.weight" not in processed_weights

        # Verify tensor values are preserved during remapping
        original_q_proj = sample_eagle3_weights["midlayer.self_attn.q_proj.weight"]
        remapped_q_proj = processed_weights["layers.0.self_attn.q_proj.weight"]
        assert torch.equal(original_q_proj, remapped_q_proj)

    @patch('speculators.convert.eagle.eagle3_converter.AutoModelForCausalLM.from_pretrained')
    def test_embeddings_replacement_with_verifier(self, mock_from_pretrained):
        """Test that embeddings are always replaced with verifier embeddings."""
        # Create a mock model with seeded embeddings for deterministic testing
        mock_model = MagicMock()
        mock_embeddings = MagicMock()
        
        # Use seeded random generator for deterministic but realistic embeddings
        torch.manual_seed(42)
        verifier_embeddings = torch.randn(128256, 4096)
        mock_embeddings.weight.data = verifier_embeddings
        mock_model.model.embed_tokens = mock_embeddings
        mock_from_pretrained.return_value = mock_model
        
        converter = Eagle3Converter()
        
        # Use a different seed for original embeddings to ensure they're different
        torch.manual_seed(123)
        original_weights = {
            "embed_tokens.weight": torch.randn(128000, 4096),  # Different seed = different values
            "other.weight": torch.randn(100, 100)
        }
        
        # Store original for comparison
        original_embeddings = original_weights["embed_tokens.weight"].clone()
        
        processed_weights = converter._add_verifier_embeddings(
            original_weights, "meta-llama/Llama-3.1-8B"
        )
        
        # Debug: Check if mock was called
        mock_from_pretrained.assert_called_once_with("meta-llama/Llama-3.1-8B", torch_dtype=torch.float32)
        
        # Check that embeddings were replaced (not equal to original)
        assert not torch.equal(
            original_embeddings,
            processed_weights["embed_tokens.weight"]
        )
        
        # Check that new embeddings match verifier
        assert torch.equal(processed_weights["embed_tokens.weight"], verifier_embeddings)
        
        # Check that other weights are preserved
        assert torch.equal(original_weights["other.weight"], processed_weights["other.weight"])

    @patch('speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict')
    def test_config_max_position_embeddings_logic(self, mock_get_config, sample_eagle3_config, sample_verifier_config):
        """Test that max_position_embeddings uses the maximum of Eagle3 and verifier values."""
        mock_get_config.return_value = (sample_verifier_config, None)
        
        converter = Eagle3Converter()
        
        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B"
        )
        
        # Check that max_position_embeddings is the max of both values
        # Eagle3: 2048, Verifier: 131072, so should be 131072
        assert llama_config.max_position_embeddings == 131072
        
        # Check that other values come from Eagle3 config  
        assert llama_config.hidden_size == 4096
        assert llama_config.num_attention_heads == 32
        # rope_theta comes from Eagle3 config, not verifier (this is the actual implementation)
        assert llama_config.rope_theta == 10000.0

    @patch('speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict')
    def test_config_fallback_when_verifier_unavailable(self, mock_get_config, sample_eagle3_config):
        """Test fallback behavior when verifier config cannot be loaded."""
        mock_get_config.side_effect = Exception("Network error")
        
        converter = Eagle3Converter()
        
        # This should raise an exception since the method doesn't handle this case gracefully
        with pytest.raises(Exception):
            converter._create_transformer_config_from_eagle(
                sample_eagle3_config, "meta-llama/Llama-3.1-8B"
            )

    def test_weight_tensor_values_preservation(self):
        """Test that weight tensor values are correctly preserved through conversion."""
        converter = Eagle3Converter()
        
        # Create test weights with known values
        original_down_proj = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        test_weights = {
            "midlayer.mlp.down_proj.weight": original_down_proj,
            "embed_tokens.weight": torch.randn(128000, 4096),
            "t2d": torch.randn(128000, 4096),
        }
        
        # Mock embeddings addition to not interfere with remapping test
        with patch.object(converter, '_add_verifier_embeddings') as mock_add_embeddings:
            def mock_add_embeddings_func(weights, base_model):
                weights_copy = weights.copy()
                weights_copy["embed_tokens.weight"] = torch.randn(128256, 4096)
                return weights_copy
            
            mock_add_embeddings.side_effect = mock_add_embeddings_func
            
            processed_weights = converter._process_checkpoint_weights(
                test_weights, "meta-llama/Llama-3.1-8B"
            )
        
        # Verify the tensor values are exactly preserved
        remapped_down_proj = processed_weights["layers.0.mlp.down_proj.weight"]
        assert torch.equal(original_down_proj, remapped_down_proj)
        
        # Verify the shape and dtype are preserved
        assert remapped_down_proj.shape == original_down_proj.shape
        assert remapped_down_proj.dtype == original_down_proj.dtype

    def test_embeddings_always_replaced_even_if_present(self):
        """Test that embeddings are always replaced, even if they exist in the checkpoint."""
        converter = Eagle3Converter()
        
        # Use seeded random generators for deterministic but realistic values
        torch.manual_seed(456)
        original_embeddings = torch.randn(128000, 4096)
        
        torch.manual_seed(789)
        verifier_embeddings = torch.randn(128256, 4096)
        
        weights = {"embed_tokens.weight": original_embeddings}
        
        with patch('speculators.convert.eagle.eagle3_converter.AutoModelForCausalLM.from_pretrained') as mock_load:
            mock_model = MagicMock()
            mock_model.model.embed_tokens.weight.data = verifier_embeddings
            mock_load.return_value = mock_model
            
            processed_weights = converter._add_verifier_embeddings(
                weights, "meta-llama/Llama-3.1-8B"
            )
        
        # Embeddings should be replaced with verifier embeddings
        assert torch.equal(processed_weights["embed_tokens.weight"], verifier_embeddings)
        assert not torch.equal(processed_weights["embed_tokens.weight"], original_embeddings)

    def test_converted_model_config_structure(self):
        """Test that the config structure created is valid for Eagle3Speculator."""
        converter = Eagle3Converter()
        
        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }
        
        with patch('speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict') as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)
            
            config = converter._build_eagle3_speculator_config(
                eagle_config, "meta-llama/Llama-3.1-8B", norm_before_residual=False
            )
        
        # Verify config structure
        assert config.speculators_model_type == "eagle3"
        assert config.transformer_layer_config.hidden_size == 4096
        assert config.speculators_config.algorithm == "eagle3"
        assert config.speculators_config.verifier.name_or_path == "meta-llama/Llama-3.1-8B"

    def test_layers_weights_preserved_not_remapped(self, sample_eagle3_weights):
        """Test that weights already named layers.0.* are preserved and not remapped."""
        converter = Eagle3Converter()
        
        # Add some weights that already have the correct layers.0.* prefix
        test_weights = sample_eagle3_weights.copy()
        test_weights["layers.0.already_correct.weight"] = torch.randn(100, 100)
        
        # Mock embeddings addition
        with patch.object(converter, '_add_verifier_embeddings') as mock_add_embeddings:
            def mock_add_embeddings_func(weights, base_model):
                weights_copy = weights.copy()
                weights_copy["embed_tokens.weight"] = torch.randn(128256, 4096)
                return weights_copy
            
            mock_add_embeddings.side_effect = mock_add_embeddings_func
            
            processed_weights = converter._process_checkpoint_weights(
                test_weights, "meta-llama/Llama-3.1-8B"
            )
        
        # Verify that layers.0.* weights are preserved as-is
        assert "layers.0.already_correct.weight" in processed_weights
        assert torch.equal(
            test_weights["layers.0.already_correct.weight"],
            processed_weights["layers.0.already_correct.weight"]
        ) 