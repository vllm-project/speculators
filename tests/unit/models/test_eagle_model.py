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
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3RMSNorm,
)
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRMSNorm
from transformers.models.granite.configuration_granite import GraniteConfig
from transformers.models.granite.modeling_granite import (
    GraniteDecoderLayer,
    GraniteRMSNorm,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from speculators import (
    SpeculatorModel,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models import EagleSpeculator, EagleSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig

# ===== Layer Types Constants =====

LAYER_TYPES: dict[str, tuple[type, type, type]] = {
    # Format: "LayerName": (LayerClass, NormClass, ConfigClass)
    "LlamaDecoderLayer": (LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig),
    "MistralDecoderLayer": (MistralDecoderLayer, MistralRMSNorm, MistralConfig),
    "Qwen3DecoderLayer": (Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3Config),
    "GemmaDecoderLayer": (GemmaDecoderLayer, GemmaRMSNorm, GemmaConfig),
    "MixtralDecoderLayer": (MixtralDecoderLayer, MixtralRMSNorm, MixtralConfig),
    "DeepseekV3DecoderLayer": (
        DeepseekV3DecoderLayer,
        DeepseekV3RMSNorm,
        DeepseekV3Config,
    ),
    "GraniteDecoderLayer": (GraniteDecoderLayer, GraniteRMSNorm, GraniteConfig),
}

LAYER_ARCHITECTURES = list(LAYER_TYPES.keys())

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


# ===== Config Helper Function =====


def create_layer_config_for_architecture(layer_architecture: str):
    config_class = LAYER_TYPES[layer_architecture][
        2
    ]  # Third element is the config class
    base_params = {
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 2048,
    }

    # Add extra parameters for specific config types
    if config_class in (MixtralConfig, DeepseekV3Config, GraniteConfig):
        base_params["num_key_value_heads"] = 12

    if config_class == MixtralConfig:
        base_params.update(
            {
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
            }
        )

    return config_class(**base_params)


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
    assert SpeculatorModel.registry is not None  # type: ignore[misc]
    assert "eagle" in SpeculatorModel.registry  # type: ignore[misc]
    assert SpeculatorModel.registry["eagle"] == EagleSpeculator  # type: ignore[misc]


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


# ===== EagleSpeculator Architecture Tests with Different Layer Types =====


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_initialization_different_layers(
    layer_architecture, sample_speculators_config
):
    """Test EagleSpeculator initialization with different layer architectures."""
    layer_config = create_layer_config_for_architecture(layer_architecture)

    # Create EagleSpeculatorConfig with the specific layer architecture
    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
    )

    # Create mock verifier for this config
    mock_verifier = MockVerifier(layer_config)

    model = EagleSpeculator(eagle_config, verifier=mock_verifier)

    assert model.config == eagle_config
    assert model.verifier == mock_verifier
    assert model.verifier_attachment_mode == "full"

    # Verify basic architecture
    assert model.embed_tokens is not None
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.rotary_emb is not None
    assert model.rotary_emb == mock_verifier.rotary_emb
    assert model.lm_head is not None
    assert model.lm_head == mock_verifier.lm_head
    assert model.fusion_fc is not None
    assert model.transformer is not None


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_architecture_different_layers(
    layer_architecture, sample_speculators_config
):
    """Test EagleSpeculator architecture with different layer types."""
    layer_config = create_layer_config_for_architecture(layer_architecture)

    # Create EagleSpeculatorConfig with the specific layer architecture
    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
    )

    # Create mock verifier for this config
    mock_verifier = MockVerifier(layer_config)

    model = EagleSpeculator(
        eagle_config, verifier=mock_verifier, verifier_attachment_mode="full"
    )

    assert isinstance(model, EagleSpeculator)
    assert isinstance(model.config, EagleSpeculatorConfig)

    # Verify embedding layer
    assert model.embed_tokens is not None
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert model.embed_tokens.weight.shape == (
        layer_config.vocab_size,
        layer_config.hidden_size,
    )

    # Verify rotary embedding
    assert model.rotary_emb is not None
    assert isinstance(model.rotary_emb, LlamaRotaryEmbedding)

    # Verify language model head
    assert model.lm_head is not None
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.shape == (
        layer_config.vocab_size,
        layer_config.hidden_size,
    )
    assert model.lm_head.bias is None

    # Verify fusion layer
    assert model.fusion_fc is not None
    assert isinstance(model.fusion_fc, nn.Linear)
    assert model.fusion_fc.weight.shape == (
        layer_config.hidden_size,
        2 * layer_config.hidden_size,
    )
    assert model.fusion_fc.bias is None

    # Verify transformer layer
    assert model.transformer is not None
    assert isinstance(model.transformer, LAYER_TYPES[layer_architecture][0])
    assert model.transformer.self_attn.config.hidden_size == layer_config.hidden_size

    # Verify no layernorms by default
    assert model.embedding_layernorm is None
    assert model.pre_lm_head_layernorm is None


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_architecture_different_layers_with_layernorms(
    layer_architecture, sample_speculators_config
):
    """Test EagleSpeculator with layernorms enabled for different decoder layers."""
    layer_config = create_layer_config_for_architecture(layer_architecture)

    # Create EagleSpeculatorConfig with layernorms enabled
    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
        layernorms=True,
        fusion_bias=True,
    )

    # Create mock verifier for this config
    mock_verifier = MockVerifier(layer_config)

    model = EagleSpeculator(
        eagle_config, verifier=mock_verifier, verifier_attachment_mode="full"
    )

    assert isinstance(model, EagleSpeculator)
    assert isinstance(model.config, EagleSpeculatorConfig)

    # Verify embedding layer
    assert model.embed_tokens is not None
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert model.embed_tokens.weight.shape == (
        layer_config.vocab_size,
        layer_config.hidden_size,
    )

    # Verify embedding layernorm
    assert model.embedding_layernorm is not None
    assert isinstance(model.embedding_layernorm, LAYER_TYPES[layer_architecture][1])
    assert model.embedding_layernorm.weight.shape == (layer_config.hidden_size,)  # type: ignore[attr-defined]

    # Verify fusion layer with bias
    assert model.fusion_fc is not None
    assert isinstance(model.fusion_fc, nn.Linear)
    assert model.fusion_fc.weight.shape == (
        layer_config.hidden_size,
        2 * layer_config.hidden_size,
    )
    assert model.fusion_fc.bias is not None

    # Verify pre-lm-head layernorm
    assert model.pre_lm_head_layernorm is not None
    assert isinstance(model.pre_lm_head_layernorm, LAYER_TYPES[layer_architecture][1])


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_from_pretrained_different_layers(
    layer_architecture, sample_speculators_config
):
    """Test EagleSpeculator from_pretrained with different layer architectures."""
    layer_config = create_layer_config_for_architecture(layer_architecture)

    # Create EagleSpeculatorConfig with the specific layer architecture
    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
    )

    # Create state dict from a detached model
    state_dict = EagleSpeculator(
        eagle_config, verifier_attachment_mode="detached"
    ).state_dict()

    # Create mock verifier for this config
    mock_verifier = MockVerifier(layer_config)

    # Load model using from_pretrained
    model = SpeculatorModel.from_pretrained(
        None,
        config=eagle_config,
        verifier=mock_verifier,
        state_dict=state_dict,
    )

    assert isinstance(model, EagleSpeculator)
    assert model.config.transformer_layer_architecture == layer_architecture
    assert model.verifier == mock_verifier
    assert model.verifier_attachment_mode == "full"
    assert model.embed_tokens == mock_verifier.embed_tokens
    assert model.rotary_emb == mock_verifier.rotary_emb
    assert model.lm_head == mock_verifier.lm_head


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_local_marshalling_different_layers(
    layer_architecture, sample_speculators_config
):
    """Test EagleSpeculator local marshalling with different layer architectures."""
    layer_config = create_layer_config_for_architecture(layer_architecture)

    # Create EagleSpeculatorConfig with the specific layer architecture
    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
    )

    # Create state dict from a detached model
    state_dict = EagleSpeculator(
        eagle_config, verifier_attachment_mode="detached"
    ).state_dict()

    # Create mock verifier for this config
    mock_verifier = MockVerifier(layer_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model = SpeculatorModel.from_pretrained(
            None,
            config=eagle_config,
            verifier=mock_verifier,
            state_dict=state_dict,
        )
        model.save_pretrained(tmpdir)  # type: ignore[attr-defined]

        # Load model from saved directory
        loaded_model = SpeculatorModel.from_pretrained(tmpdir, verifier=mock_verifier)

        assert isinstance(loaded_model, EagleSpeculator)
        assert isinstance(loaded_model.config, EagleSpeculatorConfig)
        assert loaded_model.config.transformer_layer_architecture == layer_architecture
        assert loaded_model.verifier == mock_verifier
        assert loaded_model.verifier_attachment_mode == "full"
        assert loaded_model.embed_tokens == mock_verifier.embed_tokens
        assert loaded_model.rotary_emb == mock_verifier.rotary_emb
        assert loaded_model.lm_head == mock_verifier.lm_head


# ===== EagleSpeculator Architecture Auto-Detection Tests =====


@pytest.mark.smoke
def test_eagle_speculator_auto_architecture_derivation(sample_speculators_config):
    layer_config = LlamaConfig()

    # Create config with auto architecture
    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture="auto",
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
    )

    # Create mock verifier
    mock_verifier = MockVerifier(layer_config)

    # Create model - this should work with auto architecture
    model = EagleSpeculator(eagle_config, verifier=mock_verifier)

    # Verify the model was created successfully
    assert isinstance(model, EagleSpeculator)
    # This value is set during initialization when a decoder layer class is found
    assert model.config.transformer_layer_architecture == "LlamaDecoderLayer"
    assert model.config.architectures is not None
    assert "LlamaDecoderLayer" in model.config.architectures
    assert model.verifier == mock_verifier
    assert model.verifier_attachment_mode == "full"

    # Verify the transformer layer is the correct type
    assert isinstance(model.transformer, LlamaDecoderLayer)


@pytest.mark.smoke
def test_eagle_speculator_auto_architecture_error_handling():
    # Create a custom config class that doesn't have a corresponding decoder layer
    class CustomConfig(PretrainedConfig):
        model_type = "custom"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = 32000
            self.hidden_size = 768
            self.intermediate_size = 3072
            self.num_hidden_layers = 12
            self.num_attention_heads = 12
            self.max_position_embeddings = 2048

    # This config is not in MODEL_FOR_CAUSAL_LM_MAPPING, so it should fail
    custom_config = CustomConfig()

    eagle_config = EagleSpeculatorConfig(
        transformer_layer_architecture="auto",
        transformer_layer_config=custom_config,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="test/verifier",
                architectures=["CustomForCausalLM"],
            ),
        ),
    )

    with pytest.raises(
        TypeError, match="is not a valid causal language model config class"
    ):
        EagleSpeculator(eagle_config, verifier_attachment_mode="detached")
