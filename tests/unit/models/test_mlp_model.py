"""
Unit tests for the MLPSpeculator model in the Speculators library.
"""
import copy
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import LlamaConfig

from speculators import (
    SpeculatorModel,
    SpeculatorsConfig,
    VerifierConfig
)
from speculators.models import MLPSpeculator, MLPSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


# ===== Test Helper Classes =====
class MockVerifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        embeddings = self.embed_tokens(input_ids)
        return type(
            "MockOutput",
            (),
            {"last_hidden_state": embeddings, "hidden_states": (embeddings,)},
        )()


# ===== Fixtures =====


@pytest.fixture
def sample_llama_config():
    return LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        max_position_embeddings=1024,
    )

@pytest.fixture
def sample_verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["LlamaForCausalLM"],
    )


@pytest.fixture
def sample_speculators_config(sample_verifier_config):
    return SpeculatorsConfig(
        algorithm="mlp",
        proposal_methods=[GreedyTokenProposalConfig()],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


# @pytest.fixture
# def mlp_speculator_config(sample_speculators_config):
#     return MLPSpeculatorConfig(
#         hidden_size=512,
#         intermediate_size=1024,
#         vocab_size=1000,
#         num_layers=3,
#         tie_weights=False,
#         speculators_config=sample_speculators_config,
#     )


@pytest.fixture
def mlp_speculator_config(sample_speculators_config):
    """Create a sample MLP speculator config for testing."""
    return MLPSpeculatorConfig(
        hidden_size=512,
        intermediate_size=1024,
        vocab_size=1000,
        num_layers=3,
        tie_weights=False,
        speculators_config=sample_speculators_config,
    )


@pytest.fixture
def mock_verifier(sample_llama_config):
    """Create a mock verifier model for testing."""
    return MockVerifier(sample_llama_config)


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    batch_size = 2
    seq_len = 10
    hidden_size = 512
    vocab_size = 1000

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "hidden_states": torch.randn(batch_size, seq_len, hidden_size),
        "attention_mask": torch.ones(batch_size, seq_len),
    }


# ===== MLPSpeculator Class Atttributes Tests =====


@pytest.mark.smoke
def test_mlp_class_attributes():
    assert MLPSpeculator.auto_package == "speculators.models"
    assert MLPSpeculator.registry_auto_discovery is True
    assert MLPSpeculator.config_class == MLPSpeculatorConfig
    assert MLPSpeculator.base_model_prefix == "model"
    assert MLPSpeculator.main_input_name == "input_ids"


# ===== MLPSpeculator Registry Tests =====


@pytest.mark.smoke
def test_mlp_speculator_registry():
    """Test MLPSpeculator registry."""
    assert SpeculatorModel.registry is not None
    assert "mlp" in SpeculatorModel.registry
    assert SpeculatorModel.registry["mlp"] == MLPSpeculator


@pytest.mark.smoke
def test_mlp_speculator_registered_model_class_from_config(mlp_speculator_config):
    model_class = MLPSpeculator.registered_model_class_from_config(
        mlp_speculator_config
    )
    assert model_class == MLPSpeculator


# ===== MLPSpeculator Initalization Tests =====


@pytest.mark.smoke
def test_mlp_speculator_initialization_without_verifier(mlp_speculator_config):
    """Test MLP speculator initialization."""
    mlp_speculator_config = copy.deepcopy(mlp_speculator_config)
    mlp_speculator_config.speculators_config.verifier.name_or_path = None
    model = MLPSpeculator(config=mlp_speculator_config)

    assert model.config == mlp_speculator_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"

    # Verifier-dependent layers should be None
    assert model.embed_tokens is None
    assert model.lm_head is None

    # Verify basic attributes
    assert model.n_predict == mlp_speculator_config.num_layers
    assert model.emb_dim == mlp_speculator_config.hidden_size
    assert model.inner_dim == mlp_speculator_config.intermediate_size
    assert model.vocab_size == mlp_speculator_config.vocab_size
    assert model._tie_weights == mlp_speculator_config.tie_weights

    # Verify layers are initialized
    assert hasattr(model, "emb_layers")
    assert hasattr(model, "proj_layers")
    assert hasattr(model, "head")
    assert hasattr(model, "layernorms")
    assert hasattr(model, "activation")

    # Verify layer counts
    assert len(model.emb_layers) == mlp_speculator_config.num_layers
    assert len(model.proj_layers) == mlp_speculator_config.num_layers
    assert len(model.head) == mlp_speculator_config.num_layers
    assert len(model.layernorms) == mlp_speculator_config.num_layers


@pytest.mark.smoke
def test_mlp_speculator_initialization_with_verifier_path(
    mlp_speculator_config, mock_verifier
):

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_verifier
    ):
        verifier_path = "path/to/verifier/model"
        model = MLPSpeculator(
            mlp_speculator_config,
            verifier=verifier_path,
            verifier_attachment_mode=None
        )

    assert model.config == mlp_speculator_config
    assert model.verifier == mock_verifier
    assert model.verifier_attachment_mode == "full"

    # Verifier-dependent layers should be attached
    assert model.embed_tokens is not None
    assert model.lm_head is not None
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_mlp_speculator_initialization_with_verifier_train_only(
    mlp_speculator_config, mock_verifier
):
    model = MLPSpeculator(
        config=mlp_speculator_config,
        verifier=mock_verifier,
        verifier_attachment_mode="train_only",
    )

    assert model.config == mlp_speculator_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "train_only"
    assert model.embed_tokens is not None
    assert model.lm_head is not None
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_mlp_speculator_with_verifier_detached(
    mlp_speculator_config, mock_verifier
):
    model = MLPSpeculator(
        config=mlp_speculator_config,
        verifier=mock_verifier,
        verifier_attachment_mode="detached",
    )

    assert model.config == mlp_speculator_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"
    assert model.embed_tokens is None
    assert model.lm_head is None


# ===== MLPSpeculator from_pretrained Tests =====


@pytest.mark.smoke
def test_mlp_speculator_from_pretrained_config(
    mlp_speculator_config, mock_verifier
):
    mlp_speculator_config = copy.deepcopy(mlp_speculator_config)
    state_dict = MLPSpeculator(
        mlp_speculator_config, verifier_attachment_mode="detached"
    ).state_dict()
    model = SpeculatorModel.from_pretrained(
        None,
        config=mlp_speculator_config,
        verifier=mock_verifier,
        state_dict=state_dict,
    )

    mlp_speculator_config.dtype = torch.float32
    assert isinstance(model, MLPSpeculator)
    assert model.config == mlp_speculator_config
    assert model.verifier is not None
    assert model.verifier_attachment_mode == "full"
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_mlp_speculator_from_pretrained_local_marshalling(
    mlp_speculator_config, mock_verifier
):
    mlp_speculator_config = copy.deepcopy(mlp_speculator_config)
    state_dict = MLPSpeculator(
        mlp_speculator_config,
        verifier_attachment_mode="detached"
        ).state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = MLPSpeculator.from_pretrained(
            None,
            config=mlp_speculator_config,
            verifier=mock_verifier,
            state_dict=state_dict,
        )
        model.save_pretrained(tmpdir)  # type: ignore[attr-defined]

        loaded_model = SpeculatorModel.from_pretrained(tmpdir, verifier=mock_verifier)
        mlp_speculator_config.dtype = torch.float32

        assert isinstance(loaded_model, MLPSpeculator)
        assert isinstance(loaded_model.config, MLPSpeculatorConfig)
        assert loaded_model.verifier == mock_verifier
        assert loaded_model.verifier_attachment_mode == "full"
        assert loaded_model.embed_tokens == mock_verifier.embed_tokens
        assert loaded_model.lm_head == mock_verifier.lm_head


# ===== MLPSpeculator Architecture Tests =====


@pytest.mark.smoke
def test_mlp_speculator_architecture_mlp(mlp_speculator_config, mock_verifier):
    model = MLPSpeculator(
        mlp_speculator_config,
        verifier=mock_verifier,
        verifier_attachment_mode="full",
    )
    assert isinstance(model, MLPSpeculator)
    assert isinstance(model.config, MLPSpeculatorConfig)
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.lm_head == mock_verifier.lm_head
    assert model.emb_layers is not None
    assert len(model.emb_layers) == mlp_speculator_config.num_layers
    assert model.proj_layers is not None
    assert len(model.proj_layers) == mlp_speculator_config.num_layers
    assert model.head is not None
    assert len(model.head) == mlp_speculator_config.num_layers
    assert model.layernorms is not None
    assert len(model.layernorms) == mlp_speculator_config.num_layers
