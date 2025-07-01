"""
Unit tests for the EagleSpeculator model in the Speculators library.
"""

import copy
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from speculators import (
    SpeculatorModel,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models import EagleSpeculator, EagleSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig

# ===== Test Helper Classes =====


class MockVerifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(config)
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
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=128000,
        eos_token_id=[128001, 128008, 128009],
        head_dim=128,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=131072,
        mlp_bias=False,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        pretraining_tp=1,
        rms_norm_eps=1e-5,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        rope_theta=500000.0,
        tie_word_embeddings=False,
        torch_dtype="float32",
        transformers_version="4.46.0",
        use_cache=True,
        vocab_size=128256,
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
        algorithm="eagle_v1",
        proposal_methods=[GreedyTokenProposalConfig()],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


@pytest.fixture
def eagle_speculator_config(sample_llama_config, sample_speculators_config):
    return EagleSpeculatorConfig(
        transformer_layer_config=sample_llama_config,
        speculators_config=sample_speculators_config,
    )


@pytest.fixture
def eagle_speculator_config_layernorms(sample_llama_config, sample_speculators_config):
    return EagleSpeculatorConfig(
        transformer_layer_config=sample_llama_config,
        speculators_config=sample_speculators_config,
        layernorms=True,
        fusion_bias=True,
    )


@pytest.fixture
def mock_verifier(sample_llama_config):
    return MockVerifier(sample_llama_config)


# ===== EagleSpeculator Class Attributes Tests =====


@pytest.mark.smoke
def test_eagle_speculator_class_attributes():
    assert EagleSpeculator.auto_package == "speculators.models"
    assert EagleSpeculator.registry_auto_discovery is True
    assert EagleSpeculator.config_class == EagleSpeculatorConfig
    assert EagleSpeculator.base_model_prefix == "model"
    assert EagleSpeculator.main_input_name == "input_ids"


# ===== EagleSpeculator Registry Tests =====


@pytest.mark.smoke
def test_eagle_speculator_registry():
    assert SpeculatorModel.registry is not None
    assert "eagle" in SpeculatorModel.registry
    assert SpeculatorModel.registry["eagle"] == EagleSpeculator


@pytest.mark.smoke
def test_eagle_speculator_registered_model_class_from_config(eagle_speculator_config):
    model_class = SpeculatorModel.registered_model_class_from_config(
        eagle_speculator_config
    )
    assert model_class == EagleSpeculator


# ===== EagleSpeculator Initialization Tests =====


@pytest.mark.smoke
def test_eagle_speculator_initialization_without_verifier(eagle_speculator_config):
    eagle_speculator_config = copy.deepcopy(eagle_speculator_config)
    eagle_speculator_config.speculators_config.verifier.name_or_path = None
    model = EagleSpeculator(eagle_speculator_config)

    assert model.config == eagle_speculator_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"

    # Verifier-dependent layers should be None
    assert model.embed_tokens is None
    assert model.rotary_emb is None
    assert model.lm_head is None

    # Model-specific layers should be initialized
    assert model.fusion_fc is not None
    assert model.transformer is not None
    assert isinstance(model.fusion_fc, nn.Linear)
    assert isinstance(model.transformer, LlamaDecoderLayer)


@pytest.mark.smoke
def test_eagle_speculator_initialization_with_verifier(
    eagle_speculator_config, mock_verifier
):
    model = EagleSpeculator(eagle_speculator_config, verifier=mock_verifier)

    assert model.config == eagle_speculator_config
    assert model.verifier == mock_verifier
    assert model.verifier_attachment_mode == "full"

    # Verifier-dependent layers should be attached
    assert model.embed_tokens is not None
    assert model.rotary_emb is not None
    assert model.lm_head is not None
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.rotary_emb == mock_verifier.rotary_emb
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_eagle_speculator_initialization_with_verifier_path(
    eagle_speculator_config, mock_verifier
):
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_verifier
    ):
        verifier_path = "path/to/verifier/model"
        model = EagleSpeculator(
            eagle_speculator_config,
            verifier=verifier_path,
            verifier_attachment_mode=None,
        )

        assert model.config == eagle_speculator_config
        assert model.verifier == mock_verifier
        assert model.verifier_attachment_mode == "full"
        assert model.embed_tokens is not None
        assert model.rotary_emb is not None
        assert model.lm_head is not None
        assert model.embed_tokens == mock_verifier.embed_tokens
        assert model.rotary_emb == mock_verifier.rotary_emb
        assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_eagle_speculator_initialization_with_verifier_train_only(
    eagle_speculator_config, mock_verifier
):
    model = EagleSpeculator(
        eagle_speculator_config,
        verifier=mock_verifier,
        verifier_attachment_mode="train_only",
    )

    assert model.config == eagle_speculator_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "train_only"
    assert model.embed_tokens is not None
    assert model.rotary_emb is not None
    assert model.lm_head is not None
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.rotary_emb == mock_verifier.rotary_emb
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_eagle_speculator_initialization_with_verifier_detached(
    eagle_speculator_config, mock_verifier
):
    model = EagleSpeculator(
        eagle_speculator_config,
        verifier=mock_verifier,
        verifier_attachment_mode="detached",
    )

    assert model.config == eagle_speculator_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"
    assert model.embed_tokens is None
    assert model.rotary_emb is None
    assert model.lm_head is None


# ===== EagleSpeculator from_pretrained Tests =====


@pytest.mark.smoke
def test_eagle_speculator_from_pretrained_config(
    eagle_speculator_config, mock_verifier
):
    eagle_speculator_config = copy.deepcopy(eagle_speculator_config)
    state_dict = EagleSpeculator(
        eagle_speculator_config, verifier_attachment_mode="detached"
    ).state_dict()
    model = SpeculatorModel.from_pretrained(
        None,
        config=eagle_speculator_config,
        verifier=mock_verifier,
        state_dict=state_dict,
    )

    eagle_speculator_config.torch_dtype = torch.float32
    assert isinstance(model, EagleSpeculator)
    assert model.config == eagle_speculator_config
    assert model.verifier is not None
    assert model.verifier_attachment_mode == "full"
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.rotary_emb == mock_verifier.rotary_emb
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
def test_eagle_speculator_from_pretrained_local_marshalling(
    eagle_speculator_config, mock_verifier
):
    eagle_speculator_config = copy.deepcopy(eagle_speculator_config)
    state_dict = EagleSpeculator(
        eagle_speculator_config, verifier_attachment_mode="detached"
    ).state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = SpeculatorModel.from_pretrained(
            None,
            config=eagle_speculator_config,
            verifier=mock_verifier,
            state_dict=state_dict,
        )
        model.save_pretrained(tmpdir)  # type: ignore[attr-defined]

        loaded_model = SpeculatorModel.from_pretrained(tmpdir, verifier=mock_verifier)
        eagle_speculator_config.torch_dtype = torch.float32

        assert isinstance(loaded_model, EagleSpeculator)
        assert isinstance(loaded_model.config, EagleSpeculatorConfig)
        assert (
            loaded_model.config.transformer_layer_architecture
            == eagle_speculator_config.transformer_layer_architecture
        )
        assert loaded_model.config.layernorms == eagle_speculator_config.layernorms
        assert loaded_model.config.fusion_bias == eagle_speculator_config.fusion_bias
        assert (
            loaded_model.config.speculators_config
            == eagle_speculator_config.speculators_config
        )
        assert loaded_model.verifier == mock_verifier
        assert loaded_model.verifier_attachment_mode == "full"
        assert loaded_model.embed_tokens == mock_verifier.embed_tokens
        assert loaded_model.rotary_emb == mock_verifier.rotary_emb
        assert loaded_model.lm_head == mock_verifier.lm_head


# ===== EagleSpeculator Architecture Tests =====


@pytest.mark.smoke
def test_eagle_speculator_architecture_eagle(eagle_speculator_config, mock_verifier):
    model = EagleSpeculator(
        eagle_speculator_config, verifier=mock_verifier, verifier_attachment_mode="full"
    )
    llama_config: LlamaConfig = eagle_speculator_config.transformer_layer_config

    assert isinstance(model, EagleSpeculator)
    assert isinstance(model.config, EagleSpeculatorConfig)
    assert model.embed_tokens is not None
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert model.embed_tokens.weight.shape == (
        llama_config.vocab_size,
        llama_config.hidden_size,
    )
    assert model.rotary_emb is not None
    assert isinstance(model.rotary_emb, LlamaRotaryEmbedding)
    assert model.lm_head is not None
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.shape == (
        llama_config.vocab_size,
        llama_config.hidden_size,
    )
    assert model.lm_head.bias is None
    assert model.embedding_layernorm is None
    assert model.fusion_fc is not None
    assert isinstance(model.fusion_fc, nn.Linear)
    assert model.fusion_fc.weight.shape == (
        llama_config.hidden_size,
        2 * llama_config.hidden_size,
    )
    assert model.fusion_fc.bias is None
    assert model.transformer is not None
    assert isinstance(model.transformer, LlamaDecoderLayer)
    assert model.transformer.self_attn.config.hidden_size == llama_config.hidden_size
    assert isinstance(model.transformer.input_layernorm, nn.Identity)
    assert model.pre_lm_head_layernorm is None


@pytest.mark.smoke
def test_eagle_speculator_architecture_hass(
    eagle_speculator_config_layernorms, mock_verifier
):
    model = EagleSpeculator(
        eagle_speculator_config_layernorms,
        verifier=mock_verifier,
        verifier_attachment_mode="full",
    )
    llama_config: LlamaConfig = (
        eagle_speculator_config_layernorms.transformer_layer_config
    )

    assert isinstance(model, EagleSpeculator)
    assert isinstance(model.config, EagleSpeculatorConfig)
    assert model.embed_tokens is not None
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert model.embed_tokens.weight.shape == (
        llama_config.vocab_size,
        llama_config.hidden_size,
    )
    assert model.rotary_emb is not None
    assert isinstance(model.rotary_emb, LlamaRotaryEmbedding)
    assert model.lm_head is not None
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.shape == (
        llama_config.vocab_size,
        llama_config.hidden_size,
    )
    assert model.embedding_layernorm is not None
    assert isinstance(model.embedding_layernorm, LlamaRMSNorm)
    assert model.embedding_layernorm.weight.shape == (llama_config.hidden_size,)
    assert model.fusion_fc is not None
    assert isinstance(model.fusion_fc, nn.Linear)
    assert model.fusion_fc.weight.shape == (
        llama_config.hidden_size,
        2 * llama_config.hidden_size,
    )
    assert model.fusion_fc.bias is not None
    assert model.transformer is not None
    assert isinstance(model.transformer, LlamaDecoderLayer)
    assert model.transformer.self_attn.config.hidden_size == llama_config.hidden_size
    assert isinstance(model.transformer.input_layernorm, LlamaRMSNorm)
    assert model.pre_lm_head_layernorm is not None
    assert isinstance(model.pre_lm_head_layernorm, LlamaRMSNorm)
