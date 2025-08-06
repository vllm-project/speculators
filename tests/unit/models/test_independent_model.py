"""
Unit tests for the IndependentSpeculator model in the Speculators library.
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models import (
    EagleSpeculatorConfig,
    IndependentSpeculator,
    IndependentSpeculatorConfig,
)
from speculators.proposals import GreedyTokenProposalConfig

# ===== Test Helper Classes =====


class MockDraftModel(PreTrainedModel):
    """Mock draft model for testing IndependentSpeculator."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Use simple linear layers instead of heavy transformer layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        """Simple forward pass for testing."""
        return type(
            "MockOutput",
            (),
            {
                "logits": torch.randn(
                    input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
                )
            },
        )()


class MockVerifierModel(PreTrainedModel):
    """Mock verifier model for testing IndependentSpeculator."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Use simple linear layers instead of heavy transformer layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        """Simple forward pass for testing."""
        return type(
            "MockOutput",
            (),
            {
                "logits": torch.randn(
                    input_ids.shape[0], input_ids.shape[1], self.config.vocab_size
                )
            },
        )()


# ===== Fixtures =====


@pytest.fixture
def sample_llama_config():
    """Sample LlamaConfig for testing with small dimensions for speed."""
    return LlamaConfig(
        vocab_size=1000,  # Much smaller vocab for faster tests
        hidden_size=64,  # Much smaller hidden size for faster tests
        intermediate_size=128,  # Much smaller intermediate size
        num_hidden_layers=2,  # Much fewer layers
        num_attention_heads=4,  # Fewer attention heads
        num_key_value_heads=4,  # Fewer key-value heads
        hidden_act="silu",
        max_position_embeddings=512,  # Smaller max position embeddings
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


@pytest.fixture
def sample_verifier_config():
    """Sample VerifierConfig for testing."""
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["LlamaForCausalLM"],
    )


@pytest.fixture
def sample_token_proposal_config():
    """Sample GreedyTokenProposalConfig for testing."""
    return GreedyTokenProposalConfig()


@pytest.fixture
def sample_speculators_config(sample_token_proposal_config, sample_verifier_config):
    """Sample SpeculatorsConfig for testing."""
    return SpeculatorsConfig(
        algorithm="independent",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


@pytest.fixture
def sample_speculators_config_no_verifier(sample_token_proposal_config):
    """Sample SpeculatorsConfig without verifier for testing."""
    return SpeculatorsConfig(
        algorithm="independent",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path=None,
            architectures=["LlamaForCausalLM"],
        ),
    )


@pytest.fixture
def independent_speculator_config(sample_speculators_config, sample_llama_config):
    """Sample IndependentSpeculatorConfig for testing."""
    return IndependentSpeculatorConfig.from_pretrained_config(
        pretrained_config=sample_llama_config,
        speculators_config=sample_speculators_config,
    )


@pytest.fixture
def independent_speculator_config_no_verifier(
    sample_speculators_config_no_verifier, sample_llama_config
):
    """Sample IndependentSpeculatorConfig without verifier for testing."""
    return IndependentSpeculatorConfig.from_pretrained_config(
        pretrained_config=sample_llama_config,
        speculators_config=sample_speculators_config_no_verifier,
    )


@pytest.fixture
def mock_draft_model(sample_llama_config):
    """Mock draft model for testing."""
    return MockDraftModel(sample_llama_config)


@pytest.fixture
def mock_verifier_model(sample_llama_config):
    """Mock verifier model for testing."""
    return MockVerifierModel(sample_llama_config)


# ===== IndependentSpeculator Instantiation Tests =====


@pytest.mark.smoke
def test_independent_speculator_instantiation_without_verifier(
    independent_speculator_config_no_verifier, mock_draft_model
):
    """Test IndependentSpeculator instantiation without verifier."""
    model = IndependentSpeculator(
        config=independent_speculator_config_no_verifier,
        verifier=None,
        verifier_attachment_mode="detached",
    )

    # Verify model was created successfully
    assert isinstance(model, IndependentSpeculator)
    assert model.config == independent_speculator_config_no_verifier
    assert model._draft_model is not None
    assert model._draft_model.config == independent_speculator_config_no_verifier
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"


@pytest.mark.smoke
def test_independent_speculator_instantiation_with_verifier_instance(
    independent_speculator_config, mock_draft_model, mock_verifier_model
):
    """Test IndependentSpeculator instantiation with verifier PreTrainedModel."""
    model = IndependentSpeculator(
        config=independent_speculator_config,
        verifier=mock_verifier_model,
        verifier_attachment_mode="full",
    )

    # Verify model was created successfully
    assert isinstance(model, IndependentSpeculator)
    assert model.config == independent_speculator_config
    assert isinstance(model._draft_model, LlamaForCausalLM)
    assert model.verifier == mock_verifier_model
    assert model.verifier_attachment_mode == "full"


@pytest.mark.smoke
def test_independent_speculator_instantiation_train_only_mode(
    independent_speculator_config, mock_draft_model, mock_verifier_model
):
    """Test IndependentSpeculator instantiation with train_only attachment mode."""
    model = IndependentSpeculator(
        config=independent_speculator_config,
        verifier=mock_verifier_model,
        verifier_attachment_mode="train_only",
    )

    # Verify model was created successfully
    assert isinstance(model, IndependentSpeculator)
    assert model.verifier_attachment_mode == "train_only"


@pytest.mark.smoke
def test_independent_speculator_instantiation_with_auto_verifier_from_config(
    independent_speculator_config, mock_verifier_model
):
    """Test IndependentSpeculator instantiation with verifier loaded from config."""
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[mock_verifier_model],
    ):
        model = IndependentSpeculator(
            config=independent_speculator_config,
            verifier=None,  # Should load from config
            verifier_attachment_mode="full",
        )

        # Verify model was created successfully
        assert isinstance(model, IndependentSpeculator)
        assert model.config == independent_speculator_config
        assert model.verifier == mock_verifier_model
        assert model.verifier_attachment_mode == "full"


# ===== IndependentSpeculator Error Cases Tests =====


@pytest.mark.sanity
def test_independent_speculator_instantiation_invalid_config():
    """Test IndependentSpeculator instantiation with invalid config."""
    with pytest.raises(
        ValueError, match="Attempted to initialize a IndependentSpeculator with a"
    ):
        IndependentSpeculator(
            config="invalid_config",  # type: ignore[arg-type]
            verifier=None,
            verifier_attachment_mode="detached",
        )


@pytest.mark.sanity
def test_independent_speculator_instantiation_wrong_config_type(
    sample_speculators_config,
):
    """Test IndependentSpeculator instantiation with wrong config type."""

    eagle_config = EagleSpeculatorConfig(
        transformer_layer_config=LlamaConfig(),
        speculators_config=sample_speculators_config,
    )

    with pytest.raises(
        ValueError, match="Attempted to initialize a IndependentSpeculator with a"
    ):
        IndependentSpeculator(
            config=eagle_config,  # type: ignore[arg-type]
            verifier=None,
            verifier_attachment_mode="detached",
        )
