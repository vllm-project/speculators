"""
Unit tests for the eagle model module in the Speculators library.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from transformers import PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.granite.configuration_granite import GraniteConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models import EagleSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig

LAYER_TYPES: list[tuple[str, type[PretrainedConfig]]] = [
    ("LlamaDecoderLayer", LlamaConfig),
    ("MistralDecoderLayer", MistralConfig),
    ("Qwen3DecoderLayer", Qwen3Config),
    ("GemmaDecoderLayer", GemmaConfig),
    ("MixtralDecoderLayer", MixtralConfig),
    ("DeepseekV3DecoderLayer", DeepseekV3Config),
    ("GraniteDecoderLayer", GraniteConfig),
]

LAYER_ARCHITECTURES = [arch for arch, _ in LAYER_TYPES]

# ===== Fixtures =====


@pytest.fixture
def sample_verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["LlamaForCausalLM"],
    )


@pytest.fixture
def sample_token_proposal_config():
    return GreedyTokenProposalConfig(
        speculative_tokens=5,
        verifier_accept_k=1,
        accept_tolerance=0.0,
    )


@pytest.fixture
def sample_speculators_config(sample_token_proposal_config, sample_verifier_config):
    return SpeculatorsConfig(
        algorithm="eagle",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


@pytest.fixture
def sample_llama_config():
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def sample_mistral_config():
    return MistralConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def sample_qwen3_config():
    return Qwen3Config(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def sample_gemma_config():
    return GemmaConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def sample_mixtral_config():
    return MixtralConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=2048,
        num_local_experts=8,
        num_experts_per_tok=2,
    )


@pytest.fixture
def sample_deepseek_config():
    return DeepseekV3Config(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def sample_granite_config():
    return GraniteConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def layer_configs(
    sample_llama_config,
    sample_mistral_config,
    sample_qwen3_config,
    sample_gemma_config,
    sample_mixtral_config,
    sample_deepseek_config,
    sample_granite_config,
):
    """Return a mapping of layer architectures to their configurations."""
    return {
        "LlamaDecoderLayer": sample_llama_config,
        "MistralDecoderLayer": sample_mistral_config,
        "Qwen3DecoderLayer": sample_qwen3_config,
        "GemmaDecoderLayer": sample_gemma_config,
        "MixtralDecoderLayer": sample_mixtral_config,
        "DeepseekV3DecoderLayer": sample_deepseek_config,
        "GraniteDecoderLayer": sample_granite_config,
    }


@pytest.fixture
def eagle12_config_dict():
    return {
        "speculators_model_type": "eagle",
        "architectures": ["EagleSpeculator", "LlamaDecoderLayer"],
        "transformer_layer_architecture": "LlamaDecoderLayer",
        "transformer_layer_config": {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 2048,
        },
        "layernorms": False,
        "fusion_bias": False,
        "speculators_config": {
            "algorithm": "eagle",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 5,
                    "verifier_accept_k": 1,
                    "accept_tolerance": 0.0,
                }
            ],
            "default_proposal_method": "greedy",
            "verifier": {
                "name_or_path": "test/verifier",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 768,
                "intermediate_size": 3072,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
        },
    }


@pytest.fixture
def hass_config_dict(eagle12_config_dict):
    config_dict = eagle12_config_dict.copy()
    config_dict["fusion_bias"] = True  # Key difference for HASS

    return config_dict


# ===== EagleSpeculatorConfig Tests =====


@pytest.mark.smoke
def test_eagle_speculator_config_initialization():
    """Test default initialization of EagleSpeculatorConfig."""
    config = EagleSpeculatorConfig()

    # Verify Eagle-specific defaults
    assert config.speculators_model_type == "eagle"
    assert config.architectures == ["EagleSpeculator", "LlamaDecoderLayer"]
    assert config.transformer_layer_architecture == "LlamaDecoderLayer"
    assert isinstance(config.transformer_layer_config, LlamaConfig)
    assert config.layernorms is False
    assert config.fusion_bias is False

    # Verify base class defaults
    assert config.model_type == "speculator_model"
    assert config.speculators_config is None


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_config_different_layers(layer_architecture, layer_configs):
    """Test initialization with different layer architectures."""
    layer_config = layer_configs[layer_architecture]

    config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
    )

    # Verify the correct layer architecture is set
    assert config.transformer_layer_architecture == layer_architecture
    assert layer_architecture in config.architectures
    assert "EagleSpeculator" in config.architectures
    assert config.transformer_layer_config == layer_config


@pytest.mark.smoke
def test_eagle_speculator_config_custom_initialization(
    sample_speculators_config, sample_llama_config
):
    """Test custom initialization of EagleSpeculatorConfig."""
    config = EagleSpeculatorConfig(
        architectures=["CustomEagleSpeculator"],
        transformer_layer_architecture="CustomDecoderLayer",
        transformer_layer_config=sample_llama_config,
        layernorms=True,
        fusion_bias=True,
        speculators_config=sample_speculators_config,
    )

    # Verify custom values
    assert config.speculators_model_type == "eagle"
    assert "CustomEagleSpeculator" in config.architectures
    assert "CustomDecoderLayer" in config.architectures
    assert config.transformer_layer_architecture == "CustomDecoderLayer"
    assert config.transformer_layer_config == sample_llama_config
    assert config.layernorms is True
    assert config.fusion_bias is True
    assert config.speculators_config == sample_speculators_config


@pytest.mark.smoke
@pytest.mark.parametrize(("layer_architecture", "config_class"), LAYER_TYPES)
def test_eagle_speculator_config_with_different_configs(
    layer_architecture, config_class, sample_speculators_config
):
    """Test EagleSpeculatorConfig with different transformer layer configurations."""
    layer_config = config_class(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )

    config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        speculators_config=sample_speculators_config,
    )

    # Verify the configuration
    assert config.transformer_layer_architecture == layer_architecture
    assert isinstance(config.transformer_layer_config, config_class)
    assert config.transformer_layer_config.vocab_size == 32000
    assert config.transformer_layer_config.hidden_size == 768
    assert layer_architecture in config.architectures


@pytest.mark.smoke
def test_eagle_speculator_config_base_initialization(sample_speculators_config):
    # Create EagleSpeculatorConfig with custom values
    original_config = EagleSpeculatorConfig(
        transformer_layer_architecture="TestDecoderLayer",
        layernorms=True,
        fusion_bias=True,
        speculators_config=sample_speculators_config,
    )

    # Convert to dict and validate through base class
    config_dict = original_config.model_dump()
    recreated_config = SpeculatorModelConfig.model_validate(config_dict)

    # Verify type and values preservation
    assert isinstance(recreated_config, EagleSpeculatorConfig)
    assert recreated_config.speculators_model_type == "eagle"
    assert "TestDecoderLayer" in recreated_config.architectures
    assert recreated_config.transformer_layer_architecture == "TestDecoderLayer"
    assert recreated_config.layernorms is True
    assert recreated_config.fusion_bias is True
    assert recreated_config.speculators_config == sample_speculators_config


@pytest.mark.regression
def test_eagle_speculator_config_nested_initialization():
    class ParentModel(BaseModel):
        single_config: EagleSpeculatorConfig
        config_list: list[EagleSpeculatorConfig]
        config_dict: dict[str, EagleSpeculatorConfig]

    parent = ParentModel(
        single_config=EagleSpeculatorConfig(fusion_bias=True),
        config_list=[
            EagleSpeculatorConfig(layernorms=True),
            EagleSpeculatorConfig(fusion_bias=True),
        ],
        config_dict={
            "eagle1": EagleSpeculatorConfig(layernorms=False),
            "hass": EagleSpeculatorConfig(fusion_bias=True),
        },
    )

    # Verify single config
    assert isinstance(parent.single_config, EagleSpeculatorConfig)
    assert parent.single_config.fusion_bias is True

    # Verify config list
    assert len(parent.config_list) == 2
    assert all(isinstance(c, EagleSpeculatorConfig) for c in parent.config_list)
    assert parent.config_list[0].layernorms is True
    assert parent.config_list[1].fusion_bias is True

    # Verify config dict
    assert len(parent.config_dict) == 2
    assert all(
        isinstance(c, EagleSpeculatorConfig) for c in parent.config_dict.values()
    )
    assert parent.config_dict["eagle1"].layernorms is False
    assert parent.config_dict["hass"].fusion_bias is True


@pytest.mark.smoke
def test_eagle_speculator_config_invalid_initialization():
    # Test invalid speculators_model_type
    with pytest.raises(ValidationError) as exc_info:
        EagleSpeculatorConfig(speculators_model_type="invalid")  # type: ignore[arg-type]
    assert "speculators_model_type" in str(exc_info.value)

    # Test invalid architectures type
    with pytest.raises(ValidationError) as exc_info:
        EagleSpeculatorConfig(architectures="not_a_list")  # type: ignore[arg-type]
    assert "architectures" in str(exc_info.value)

    # Test invalid transformer_layer_architecture type
    with pytest.raises(ValidationError) as exc_info:
        EagleSpeculatorConfig(transformer_layer_architecture=123)  # type: ignore[arg-type]
    assert "transformer_layer_architecture" in str(exc_info.value)

    # Test invalid layernorms type
    with pytest.raises(ValidationError) as exc_info:
        EagleSpeculatorConfig(layernorms="not_a_bool")  # type: ignore[arg-type]
    assert "layernorms" in str(exc_info.value)

    # Test invalid fusion_bias type
    with pytest.raises(ValidationError) as exc_info:
        EagleSpeculatorConfig(fusion_bias="not_a_bool")  # type: ignore[arg-type]
    assert "fusion_bias" in str(exc_info.value)


@pytest.mark.smoke
def test_eagle_speculator_config_auto_registry():
    registered_classes = SpeculatorModelConfig.registered_classes()
    class_names = [cls.__name__ for cls in registered_classes]

    # Verify EagleSpeculatorConfig is registered
    assert "EagleSpeculatorConfig" in class_names

    # Verify registry key mapping
    assert SpeculatorModelConfig.registry is not None
    assert "eagle" in SpeculatorModelConfig.registry
    assert SpeculatorModelConfig.registry["eagle"] == EagleSpeculatorConfig


@pytest.mark.smoke
def test_eagle_speculator_config_marshalling(sample_speculators_config):
    original_config = EagleSpeculatorConfig(
        transformer_layer_architecture="TestDecoderLayer",
        layernorms=True,
        fusion_bias=True,
        speculators_config=sample_speculators_config,
    )

    # Test model_dump()
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "eagle"
    assert "TestDecoderLayer" in config_dict["architectures"]
    assert config_dict["layernorms"] is True
    assert config_dict["fusion_bias"] is True

    # Test model_validate() on base class
    recreated_base = SpeculatorModelConfig.model_validate(config_dict)
    assert isinstance(recreated_base, EagleSpeculatorConfig)
    assert recreated_base.transformer_layer_architecture == "TestDecoderLayer"
    assert recreated_base.layernorms is True
    assert recreated_base.fusion_bias is True

    # Test model_validate() on derived class
    recreated_derived = EagleSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, EagleSpeculatorConfig)
    assert recreated_derived.transformer_layer_architecture == "TestDecoderLayer"
    assert recreated_derived.layernorms is True
    assert recreated_derived.fusion_bias is True


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_config_marshalling_different_layers(
    layer_architecture, layer_configs, sample_speculators_config
):
    """Test marshalling with different layer architectures."""
    layer_config = layer_configs[layer_architecture]

    original_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        layernorms=True,
        fusion_bias=True,
        speculators_config=sample_speculators_config,
    )

    # Test model_dump()
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "eagle"
    assert config_dict["transformer_layer_architecture"] == layer_architecture
    assert layer_architecture in config_dict["architectures"]

    # Test model_validate() on base class
    recreated_base = SpeculatorModelConfig.model_validate(config_dict)
    assert isinstance(recreated_base, EagleSpeculatorConfig)
    assert recreated_base.transformer_layer_architecture == layer_architecture
    assert recreated_base.layernorms is True
    assert recreated_base.fusion_bias is True

    # Test model_validate() roundtrip
    recreated_config = EagleSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_config, EagleSpeculatorConfig)
    assert recreated_config.transformer_layer_architecture == layer_architecture
    assert recreated_config.layernorms is True
    assert recreated_config.fusion_bias is True


@pytest.mark.smoke
def test_eagle_speculator_config_model_validator():
    config1 = EagleSpeculatorConfig(transformer_layer_architecture="CustomDecoderLayer")
    assert "CustomDecoderLayer" in config1.architectures
    assert "EagleSpeculator" in config1.architectures

    # Test with custom architectures already containing the layer
    config2 = EagleSpeculatorConfig(
        architectures=["CustomSpeculator", "CustomDecoderLayer"],
        transformer_layer_architecture="CustomDecoderLayer",
    )
    # Should not duplicate
    architecture_count = config2.architectures.count("CustomDecoderLayer")
    assert architecture_count == 1
    assert "CustomSpeculator" in config2.architectures

    # Test with custom architectures not containing the layer
    config3 = EagleSpeculatorConfig(
        architectures=["CustomSpeculator"],
        transformer_layer_architecture="NewDecoderLayer",
    )
    assert "CustomSpeculator" in config3.architectures
    assert "NewDecoderLayer" in config3.architectures


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_config_model_validator_different_layers(layer_architecture):
    """Test model validator with different layer architectures."""
    config = EagleSpeculatorConfig(transformer_layer_architecture=layer_architecture)

    # Verify the layer architecture is added to architectures
    assert layer_architecture in config.architectures
    assert "EagleSpeculator" in config.architectures

    # Verify no duplication occurs
    architecture_count = config.architectures.count(layer_architecture)
    assert architecture_count == 1


# # ====== EagleSpeculatorConfig Eagle 1 / Eagle 2 Tests ======


@pytest.mark.smoke
def test_eagle_speculator_config_eagle12_backwards_compatibility(eagle12_config_dict):
    config_derived = EagleSpeculatorConfig.model_validate(eagle12_config_dict)
    assert isinstance(config_derived, EagleSpeculatorConfig)
    assert config_derived.speculators_model_type == "eagle"
    assert "LlamaDecoderLayer" in config_derived.architectures
    assert config_derived.transformer_layer_architecture == "LlamaDecoderLayer"
    assert config_derived.layernorms is False
    assert config_derived.fusion_bias is False
    assert config_derived.speculators_config.algorithm == "eagle"

    # Test loading with base SpeculatorModelConfig.model_validate
    config_base = SpeculatorModelConfig.model_validate(eagle12_config_dict)
    assert isinstance(config_base, EagleSpeculatorConfig)
    assert config_base.speculators_model_type == "eagle"
    assert config_base.transformer_layer_architecture == "LlamaDecoderLayer"
    assert config_base.layernorms is False
    assert config_base.fusion_bias is False
    assert config_base.speculators_config.algorithm == "eagle"


@pytest.mark.smoke
@pytest.mark.parametrize(("layer_architecture", "config_class"), LAYER_TYPES)
def test_eagle_speculator_config_backwards_compatibility_different_layers(
    layer_architecture, config_class, sample_speculators_config
):
    """Test backwards compatibility with different layer architectures."""
    model_type = layer_architecture.lower().replace("decoderlayer", "")
    if model_type == "deepseekv3":
        model_type = "deepseek_v3"

    config_dict = {
        "speculators_model_type": "eagle",
        "architectures": ["EagleSpeculator", layer_architecture],
        "transformer_layer_architecture": layer_architecture,
        "transformer_layer_config": {
            "model_type": model_type,
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 2048,
        },
        "layernorms": False,
        "fusion_bias": False,
        "speculators_config": sample_speculators_config.model_dump(),
    }

    config = EagleSpeculatorConfig.model_validate(config_dict)
    assert isinstance(config, EagleSpeculatorConfig)
    assert config.speculators_model_type == "eagle"
    assert config.transformer_layer_architecture == layer_architecture
    assert layer_architecture in config.architectures
    assert isinstance(config.transformer_layer_config, config_class)


@pytest.mark.smoke
def test_eagle_speculator_config_eagle12_dict_marshalling(eagle12_config_dict):
    original_config = EagleSpeculatorConfig.model_validate(eagle12_config_dict)

    # Convert to dict with model_dump
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "eagle"
    assert config_dict["fusion_bias"] is False

    # Load with from_dict on base class
    recreated_base = SpeculatorModelConfig.from_dict(config_dict)
    assert isinstance(recreated_base, EagleSpeculatorConfig)
    assert recreated_base.fusion_bias is False
    assert recreated_base.layernorms is False
    assert recreated_base.transformer_layer_architecture == "LlamaDecoderLayer"

    # Load with from_dict on derived class (should work through inheritance)
    recreated_derived = EagleSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, EagleSpeculatorConfig)
    assert recreated_derived.fusion_bias is False
    assert recreated_derived.layernorms is False
    assert recreated_derived.transformer_layer_architecture == "LlamaDecoderLayer"


@pytest.mark.smoke
def test_eagle_speculator_config_eagle12_from_pretrained_local_marshalling(
    eagle12_config_dict,
):
    original_config = EagleSpeculatorConfig.model_validate(eagle12_config_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save with save_pretrained
        original_config.save_pretrained(temp_path)

        # Verify config.json was created
        config_file = temp_path / "config.json"
        assert config_file.exists()

        # Load with from_pretrained on base class
        loaded_base = SpeculatorModelConfig.from_pretrained(temp_path)
        assert isinstance(loaded_base, EagleSpeculatorConfig)
        assert loaded_base.speculators_model_type == "eagle"
        assert loaded_base.fusion_bias is False
        assert loaded_base.layernorms is False
        assert loaded_base.transformer_layer_architecture == "LlamaDecoderLayer"

        # Load with from_pretrained on derived class
        loaded_derived = EagleSpeculatorConfig.from_pretrained(temp_path)
        assert isinstance(loaded_derived, EagleSpeculatorConfig)
        assert loaded_derived.speculators_model_type == "eagle"
        assert loaded_derived.fusion_bias is False
        assert loaded_derived.layernorms is False
        assert loaded_derived.transformer_layer_architecture == "LlamaDecoderLayer"


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_config_from_pretrained_different_layers(
    layer_architecture, layer_configs, sample_speculators_config
):
    """Test from_pretrained with different layer architectures."""
    layer_config = layer_configs[layer_architecture]

    original_config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        layernorms=False,
        fusion_bias=False,
        speculators_config=sample_speculators_config,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save with save_pretrained
        original_config.save_pretrained(temp_path)

        # Verify config.json was created
        config_file = temp_path / "config.json"
        assert config_file.exists()

        # Load with from_pretrained
        loaded_config = EagleSpeculatorConfig.from_pretrained(temp_path)
        assert isinstance(loaded_config, EagleSpeculatorConfig)
        assert loaded_config.speculators_model_type == "eagle"
        assert loaded_config.transformer_layer_architecture == layer_architecture
        assert layer_architecture in loaded_config.architectures


# ====== EagleSpeculatorConfig HASS Tests ======


@pytest.mark.smoke
def test_eagle_speculator_config_hass_backwards_compatibility(hass_config_dict):
    config_derived = EagleSpeculatorConfig.model_validate(hass_config_dict)
    assert isinstance(config_derived, EagleSpeculatorConfig)
    assert config_derived.speculators_model_type == "eagle"
    assert "LlamaDecoderLayer" in config_derived.architectures
    assert config_derived.transformer_layer_architecture == "LlamaDecoderLayer"
    assert config_derived.layernorms is False
    assert config_derived.fusion_bias is True  # Key difference for HASS
    assert config_derived.speculators_config.algorithm == "eagle"

    # Test loading with base SpeculatorModelConfig.model_validate
    config_base = SpeculatorModelConfig.model_validate(hass_config_dict)
    assert isinstance(config_base, EagleSpeculatorConfig)
    assert config_base.speculators_model_type == "eagle"
    assert config_base.transformer_layer_architecture == "LlamaDecoderLayer"
    assert config_base.layernorms is False
    assert config_base.fusion_bias is True  # Key difference for HASS
    assert config_base.speculators_config.algorithm == "eagle"


@pytest.mark.smoke
@pytest.mark.parametrize("layer_architecture", LAYER_ARCHITECTURES)
def test_eagle_speculator_config_hass_different_layers(
    layer_architecture, layer_configs
):
    """Test HASS configuration with different layer architectures."""
    layer_config = layer_configs[layer_architecture]

    config = EagleSpeculatorConfig(
        transformer_layer_architecture=layer_architecture,
        transformer_layer_config=layer_config,
        layernorms=False,
        fusion_bias=True,  # Key difference for HASS
    )

    assert isinstance(config, EagleSpeculatorConfig)
    assert config.speculators_model_type == "eagle"
    assert config.transformer_layer_architecture == layer_architecture
    assert layer_architecture in config.architectures
    assert config.fusion_bias is True  # Key difference for HASS
    assert config.layernorms is False


@pytest.mark.smoke
def test_eagle_speculator_config_hass_dict_marshalling(hass_config_dict):
    original_config = EagleSpeculatorConfig.model_validate(hass_config_dict)

    # Convert to dict with model_dump
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "eagle"
    assert config_dict["fusion_bias"] is True  # Key difference for HASS

    # Load with from_dict on base class
    recreated_base = SpeculatorModelConfig.from_dict(config_dict)
    assert isinstance(recreated_base, EagleSpeculatorConfig)
    assert recreated_base.fusion_bias is True  # Key difference for HASS
    assert recreated_base.layernorms is False
    assert recreated_base.transformer_layer_architecture == "LlamaDecoderLayer"

    # Load with from_dict on derived class (should work through inheritance)
    recreated_derived = EagleSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, EagleSpeculatorConfig)
    assert recreated_derived.fusion_bias is True  # Key difference for HASS
    assert recreated_derived.layernorms is False
    assert recreated_derived.transformer_layer_architecture == "LlamaDecoderLayer"


@pytest.mark.smoke
def test_eagle_speculator_config_hass_from_pretrained_local_marshalling(
    hass_config_dict,
):
    original_config = EagleSpeculatorConfig.model_validate(hass_config_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save with save_pretrained
        original_config.save_pretrained(temp_path)

        # Verify config.json was created
        config_file = temp_path / "config.json"
        assert config_file.exists()

        # Load with from_pretrained on base class
        loaded_base = SpeculatorModelConfig.from_pretrained(temp_path)
        assert isinstance(loaded_base, EagleSpeculatorConfig)
        assert loaded_base.speculators_model_type == "eagle"
        assert loaded_base.fusion_bias is True  # Key difference for HASS
        assert loaded_base.layernorms is False
        assert loaded_base.transformer_layer_architecture == "LlamaDecoderLayer"

        # Load with from_pretrained on derived class
        loaded_derived = EagleSpeculatorConfig.from_pretrained(temp_path)
        assert isinstance(loaded_derived, EagleSpeculatorConfig)
        assert loaded_derived.speculators_model_type == "eagle"
        assert loaded_derived.fusion_bias is True  # Key difference for HASS
        assert loaded_derived.layernorms is False
        assert loaded_derived.transformer_layer_architecture == "LlamaDecoderLayer"
