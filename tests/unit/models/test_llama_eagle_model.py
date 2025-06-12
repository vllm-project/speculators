"""Unit tests for LlamaEagleSpeculator model."""

import pytest
import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.models.llama_eagle import (
    LlamaEagleSpeculator,
    LlamaEagleSpeculatorConfig,
)


class TestLlamaEagleSpeculator:
    """Test suite for LlamaEagleSpeculator model."""

    @pytest.fixture
    def eagle_v1_config(self):
        """Create a test EAGLE v1 configuration."""
        llama_config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=512,
            mlp_bias=False,
        )
        return LlamaEagleSpeculatorConfig(
            eagle_variant="eagle",
            num_hidden_layers=2,
            llama_decoder_layer_config=llama_config,
        )

    @pytest.fixture
    def hass_config(self):
        """Create a test HASS configuration."""
        llama_config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=512,
            mlp_bias=False,
        )
        return LlamaEagleSpeculatorConfig(
            eagle_variant="hass",
            num_hidden_layers=2,
            llama_decoder_layer_config=llama_config,
        )

    def test_model_initialization_eagle_v1(self, eagle_v1_config):
        """Test EAGLE v1 model initialization."""
        model = LlamaEagleSpeculator(eagle_v1_config)

        # Check basic attributes
        assert model.config == eagle_v1_config
        assert model.vocab_size == 1000

        # Check fusion layer
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.in_features == 256 * 2  # hidden_size * 2
        assert model.fc.out_features == 256
        assert model.fc.bias is None  # No bias for EAGLE v1

        # Check embeddings
        assert isinstance(model.embed_tokens, nn.Embedding)
        assert model.embed_tokens.num_embeddings == 1000
        assert model.embed_tokens.embedding_dim == 256

        # Check that extra layernorms are not created
        assert not hasattr(model, "extra_layernorms")

        # Check first layer normalization is Identity
        assert isinstance(model.layers[0].input_layernorm, nn.Identity)

        # Check decoder layers
        assert len(model.layers) == 2

    def test_model_initialization_hass(self, hass_config):
        """Test HASS model initialization."""
        model = LlamaEagleSpeculator(hass_config)

        # Check fusion layer has bias
        assert model.fc.bias is not None

        # Check extra layernorms are created
        assert hasattr(model, "extra_layernorms")
        assert "post_embedding" in model.extra_layernorms
        assert "pre_lm_head" in model.extra_layernorms

    def test_forward_pass_shapes(self, eagle_v1_config):
        """Test forward pass output shapes."""
        model = LlamaEagleSpeculator(eagle_v1_config)
        model.eval()

        batch_size = 2
        seq_length = 10

        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        hidden_states = torch.randn(batch_size, seq_length, 256)

        with torch.no_grad():
            logits = model(input_ids=input_ids, hidden_states=hidden_states)

        # Check output shape
        assert logits.shape == (batch_size, seq_length, 1000)

    def test_forward_pass_with_attention_mask(self, eagle_v1_config):
        """Test forward pass with attention mask."""
        model = LlamaEagleSpeculator(eagle_v1_config)
        model.eval()

        batch_size = 2
        seq_length = 10

        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        hidden_states = torch.randn(batch_size, seq_length, 256)
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[0, 5:] = 0  # Mask out last 5 tokens for first sample

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )

        assert logits.shape == (batch_size, seq_length, 1000)

    def test_gradient_flow(self, eagle_v1_config):
        """Test that gradients flow through the model."""
        model = LlamaEagleSpeculator(eagle_v1_config)
        model.train()

        # Create inputs
        input_ids = torch.randint(0, 1000, (1, 5))
        hidden_states = torch.randn(1, 5, 256, requires_grad=True)

        # Forward pass
        logits = model(input_ids=input_ids, hidden_states=hidden_states)

        # Create dummy loss
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert model.fc.weight.grad is not None
        assert model.embed_tokens.weight.grad is not None

    def test_get_set_input_embeddings(self, eagle_v1_config):
        """Test get/set input embeddings methods."""
        model = LlamaEagleSpeculator(eagle_v1_config)

        # Get embeddings
        embeddings = model.get_input_embeddings()
        assert embeddings is model.embed_tokens

        # Set new embeddings
        new_embeddings = nn.Embedding(2000, 256)
        model.set_input_embeddings(new_embeddings)
        assert model.get_input_embeddings() is new_embeddings
