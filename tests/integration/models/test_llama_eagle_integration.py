"""Integration tests for LlamaEagleSpeculator model."""

import tempfile
from pathlib import Path

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.llama_eagle import (
    LlamaEagleSpeculator,
    LlamaEagleSpeculatorConfig,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig


class TestLlamaEagleIntegration:
    """Integration tests for LlamaEagleSpeculator."""

    @pytest.fixture
    def speculators_config(self):
        """Create a SpeculatorsConfig for testing."""
        # Create a minimal token proposal config
        proposal_config = GreedyTokenProposalConfig()

        # Create a verifier config
        verifier_config = VerifierConfig(
            name_or_path="meta-llama/Llama-2-7b-hf",
            architectures=["LlamaForCausalLM"],
            hidden_size=2048,
            intermediate_size=5632,
            vocab_size=32000,
            max_position_embeddings=2048,
            bos_token_id=1,
            eos_token_id=2,
        )

        return SpeculatorsConfig(
            algorithm="eagle",
            proposal_methods=[proposal_config],
            default_proposal_method="greedy",
            verifier=verifier_config,
        )

    @pytest.fixture
    def eagle_config(self, speculators_config):
        """Create a realistic EAGLE v1 configuration."""
        llama_config = LlamaConfig(
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=32000,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mlp_bias=False,
        )
        return LlamaEagleSpeculatorConfig(
            eagle_variant="eagle",
            num_hidden_layers=1,
            llama_decoder_layer_config=llama_config,
            speculators_config=speculators_config,
        )

    @pytest.fixture
    def hass_config(self, speculators_config):
        """Create a realistic HASS configuration."""
        llama_config = LlamaConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=128000,
            eos_token_id=[128001, 128008, 128009],
            mlp_bias=False,
        )
        return LlamaEagleSpeculatorConfig(
            eagle_variant="hass",
            num_hidden_layers=1,
            llama_decoder_layer_config=llama_config,
            speculators_config=speculators_config,
        )

    def test_save_and_load_config(self, eagle_config):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save config
            eagle_config.save_pretrained(save_path)

            # Check config.json exists
            config_path = save_path / "config.json"
            assert config_path.exists()

            # Load config
            loaded_config = LlamaEagleSpeculatorConfig.from_pretrained(save_path)

            # Check attributes match
            assert loaded_config.eagle_variant == eagle_config.eagle_variant
            assert loaded_config.fc_bias == eagle_config.fc_bias
            assert (
                loaded_config.use_extra_layernorms == eagle_config.use_extra_layernorms
            )
            assert loaded_config.hidden_size == eagle_config.hidden_size
            assert loaded_config.vocab_size == eagle_config.vocab_size

    def test_save_and_load_model(self, eagle_config):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Create and save model
            model = LlamaEagleSpeculator(eagle_config)
            model.save_pretrained(save_path)

            # Check files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "model.safetensors").exists()

            # Load model
            loaded_model = LlamaEagleSpeculator.from_pretrained(save_path)

            # Check model architecture matches
            assert isinstance(loaded_model, LlamaEagleSpeculator)
            assert loaded_model.config.eagle_variant == "eagle"

            # Test forward pass with loaded model
            input_ids = torch.randint(0, 1000, (1, 10))
            hidden_states = torch.randn(1, 10, 2048)

            with torch.no_grad():
                original_logits = model(
                    input_ids=input_ids, hidden_states=hidden_states
                )
                loaded_logits = loaded_model(
                    input_ids=input_ids, hidden_states=hidden_states
                )

            # Check outputs match (approximately, potential numerical differences)
            assert torch.allclose(original_logits, loaded_logits, atol=1e-5)

    def test_model_with_different_hidden_states_inputs(self, eagle_config):
        """Test model with different hidden states input configurations."""
        # Modify config to use different hidden states
        eagle_config.inputs = ["input_ids", "hidden_states[-2]"]
        model = LlamaEagleSpeculator(eagle_config)
        model.eval()

        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 10))
        hidden_states = torch.randn(1, 10, 2048)

        with torch.no_grad():
            logits = model(input_ids=input_ids, hidden_states=hidden_states)

        assert logits.shape == (1, 10, 32000)

    def test_hass_extra_layernorms(self, hass_config):
        """Test HASS model with extra layernorms."""
        model = LlamaEagleSpeculator(hass_config)

        # Check extra layernorms exist
        assert hasattr(model, "extra_layernorms")
        assert "post_embedding" in model.extra_layernorms
        assert "pre_lm_head" in model.extra_layernorms

        # Test that they're used in forward pass
        input_ids = torch.randint(0, 1000, (1, 5))
        hidden_states = torch.randn(1, 5, 4096)

        # Hook to check if layernorms are called
        post_embedding_called = False
        pre_lm_head_called = False

        def hook_post_embedding(module, input_, output):
            nonlocal post_embedding_called
            post_embedding_called = True

        def hook_pre_lm_head(module, input_, output):
            nonlocal pre_lm_head_called
            pre_lm_head_called = True

        model.extra_layernorms["post_embedding"].register_forward_hook(
            hook_post_embedding
        )
        model.extra_layernorms["pre_lm_head"].register_forward_hook(hook_pre_lm_head)

        with torch.no_grad():
            model(input_ids=input_ids, hidden_states=hidden_states)

        assert post_embedding_called
        assert pre_lm_head_called
