"""
Unit tests for the MLP speculator config module in the Speculators library.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
import torch
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
from speculators.models import MLPSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


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
        algorithm="mlp",
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
def mlp_config_dict():
    return {
        "speculators_model_type": "mlp",
        "architectures": ["MLPSpeculator"],
        "dtype": "bfloat16",
        "inputs": ["input_embeddings", "hidden_states[-1]"],
        "inputs_hidden_states_normalized": False,
        "hidden_size": 4096,
        "intermediate_size": 4096,
        "vocab_size": 128256,
        "num_layers": 5,
        "tie_weights": True,
        "speculators_config": {
            "algorithm": "mlp",
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
                "hidden_size": 4096,
                "intermediate_size": 4096,
                "vocab_size": 128256,
                "max_position_embeddings": 2048,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
        },
    }


# ===== Config Classes =====

LAYER_TYPES: list[tuple[str, type[PretrainedConfig]]] = [
    ("LlamaDecoderLayer", LlamaConfig),
    ("MistralDecoderLayer", MistralConfig),
    ("Qwen3DecoderLayer", Qwen3Config),
    ("GemmaDecoderLayer", GemmaConfig),
    ("MixtralDecoderLayer", MixtralConfig),
    ("DeepseekV3DecoderLayer", DeepseekV3Config),
    ("GraniteDecoderLayer", GraniteConfig),
]


def create_layer_config(config_class: type[PretrainedConfig]) -> PretrainedConfig:
    """Create a config instance for the given config class with standard parameters."""
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

    return config_class(**base_params)  # type: ignore[arg-type]


# ===== MLPSpeculatorConfig Tests =====


@pytest.mark.smoke
def test_mlp_speculator_config_initialization():
    """Test default initialization of MLPSpeculatorConfig."""
    config = MLPSpeculatorConfig()

    # Verify MLP-specific defaults
    assert config.speculators_model_type == "mlp"
    assert config.architectures == ["MLPSpeculator"]
    assert config.dtype == torch.bfloat16
    assert config.inputs == ["input_embeddings", "hidden_states[-1]"]
    assert config.inputs_hidden_states_normalized is False
    assert config.hidden_size == 4096
    assert config.intermediate_size == 4096
    assert config.vocab_size == 128256
    assert config.num_layers == 5
    assert config.tie_weights is True

    # Verify base class defaults
    assert config.model_type == "speculator_model"
    assert config.speculators_config is None


@pytest.mark.smoke
def test_mlp_speculator_config_custom_initialization(
    sample_speculators_config, sample_llama_config
):
    """Test custom initialization of MLPSpeculatorConfig."""
    config = MLPSpeculatorConfig(
        architectures=["CustomMLPSpeculator"],
        dtype="bfloat16",
        inputs=["custom_input"],
        inputs_hidden_states_normalized=True,
        hidden_size=2048,
        intermediate_size=1024,
        vocab_size=50000,
        num_layers=3,
        tie_weights=False,
        speculators_config=sample_speculators_config,
    )

    # Verify custom values
    assert config.speculators_model_type == "mlp"
    assert "CustomMLPSpeculator" in config.architectures
    assert config.dtype == torch.bfloat16
    assert config.inputs == ["custom_input"]
    assert config.inputs_hidden_states_normalized is True
    assert config.hidden_size == 2048
    assert config.intermediate_size == 1024
    assert config.vocab_size == 50000
    assert config.num_layers == 3
    assert config.tie_weights is False
    assert config.speculators_config == sample_speculators_config


@pytest.mark.smoke
def test_mlp_speculator_config_base_initialization(sample_speculators_config):
    # Create MLPSpeculatorConfig with custom values
    original_config = MLPSpeculatorConfig(
        hidden_size=2048,
        intermediate_size=1024,
        num_layers=3,
        tie_weights=False,
        speculators_config=sample_speculators_config,
    )

    # Convert to dict and validate through base class
    config_dict = original_config.model_dump()
    recreated_config = SpeculatorModelConfig.model_validate(config_dict)

    # Verify type and values preservation
    assert isinstance(recreated_config, MLPSpeculatorConfig)
    assert recreated_config.speculators_model_type == "mlp"
    assert recreated_config.hidden_size == 2048
    assert recreated_config.intermediate_size == 1024
    assert recreated_config.num_layers == 3
    assert recreated_config.tie_weights is False
    assert recreated_config.speculators_config == sample_speculators_config


@pytest.mark.regression
def test_mlp_speculator_config_nested_initialization():
    class ParentModel(BaseModel):
        single_config: MLPSpeculatorConfig
        config_list: list[MLPSpeculatorConfig]
        config_dict: dict[str, MLPSpeculatorConfig]

    parent = ParentModel(
        single_config=MLPSpeculatorConfig(tie_weights=True),
        config_list=[
            MLPSpeculatorConfig(num_layers=3),
            MLPSpeculatorConfig(tie_weights=True),
        ],
        config_dict={
            "mlp1": MLPSpeculatorConfig(num_layers=3),
            "mlp2": MLPSpeculatorConfig(tie_weights=True),
        },
    )

    # Verify single config
    assert isinstance(parent.single_config, MLPSpeculatorConfig)
    assert parent.single_config.tie_weights is True

    # Verify config list
    assert len(parent.config_list) == 2
    assert all(isinstance(c, MLPSpeculatorConfig) for c in parent.config_list)
    assert parent.config_list[0].num_layers == 3
    assert parent.config_list[1].tie_weights is True

    # Verify config dict
    assert len(parent.config_dict) == 2
    assert all(
        isinstance(c, MLPSpeculatorConfig) for c in parent.config_dict.values()
    )
    assert parent.config_dict["mlp1"].num_layers == 3
    assert parent.config_dict["mlp2"].tie_weights is True


@pytest.mark.smoke
def test_mlp_speculator_config_invalid_initialization():
    # Test invalid speculators_model_type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(speculators_model_type="invalid")  # type: ignore[arg-type]
    assert "speculators_model_type" in str(exc_info.value)

    # Test invalid architectures type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(architectures="not_a_list")  # type: ignore[arg-type]
    assert "architectures" in str(exc_info.value)

    # Test invalid dtype type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(dtype=123)  # type: ignore[arg-type]
    assert "dtype" in str(exc_info.value)

    # Test invalid inputs type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(inputs="not_a_list")  # type: ignore[arg-type]
    assert "inputs" in str(exc_info.value)

    # Test invalid hidden_size type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(hidden_size="not_an_int")  # type: ignore[arg-type]
    assert "hidden_size" in str(exc_info.value)

    # Test invalid num_layers type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(num_layers="not_an_int")  # type: ignore[arg-type]
    assert "num_layers" in str(exc_info.value)

    # Test invalid tie_weights type
    with pytest.raises(ValidationError) as exc_info:
        MLPSpeculatorConfig(tie_weights="not_a_bool")  # type: ignore[arg-type]
    assert "tie_weights" in str(exc_info.value)


@pytest.mark.smoke
def test_mlp_speculator_config_auto_registry():
    registered_classes = SpeculatorModelConfig.registered_classes()
    class_names = [cls.__name__ for cls in registered_classes]

    # Verify MLPSpeculatorConfig is registered
    assert "MLPSpeculatorConfig" in class_names

    # Verify registry key mapping
    assert SpeculatorModelConfig.registry is not None
    assert "mlp" in SpeculatorModelConfig.registry
    assert SpeculatorModelConfig.registry["mlp"] == MLPSpeculatorConfig


@pytest.mark.smoke
def test_mlp_speculator_config_marshalling(sample_speculators_config):
    original_config = MLPSpeculatorConfig(
        hidden_size=2048,
        intermediate_size=1024,
        num_layers=3,
        tie_weights=False,
        speculators_config=sample_speculators_config,
    )

    # Test model_dump()
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "mlp"
    assert config_dict["hidden_size"] == 2048
    assert config_dict["intermediate_size"] == 1024
    assert config_dict["num_layers"] == 3
    assert config_dict["tie_weights"] is False

    # Test model_validate() on base class
    recreated_base = SpeculatorModelConfig.model_validate(config_dict)
    assert isinstance(recreated_base, MLPSpeculatorConfig)
    assert recreated_base.hidden_size == 2048
    assert recreated_base.intermediate_size == 1024
    assert recreated_base.num_layers == 3
    assert recreated_base.tie_weights is False

    # Test model_validate() on derived class
    recreated_derived = MLPSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, MLPSpeculatorConfig)
    assert recreated_derived.hidden_size == 2048
    assert recreated_derived.intermediate_size == 1024
    assert recreated_derived.num_layers == 3
    assert recreated_derived.tie_weights is False


@pytest.mark.smoke
def test_mlp_speculator_config_backwards_compatibility(mlp_config_dict):
    config_derived = MLPSpeculatorConfig.model_validate(mlp_config_dict)
    assert isinstance(config_derived, MLPSpeculatorConfig)
    assert config_derived.speculators_model_type == "mlp"
    assert config_derived.dtype == torch.bfloat16
    assert config_derived.inputs == ["input_embeddings", "hidden_states[-1]"]
    assert config_derived.inputs_hidden_states_normalized is False
    assert config_derived.hidden_size == 4096
    assert config_derived.intermediate_size == 4096
    assert config_derived.vocab_size == 128256
    assert config_derived.num_layers == 5
    assert config_derived.tie_weights is True
    assert config_derived.speculators_config.algorithm == "mlp"

    # Test loading with base SpeculatorModelConfig.model_validate
    config_base = SpeculatorModelConfig.model_validate(mlp_config_dict)
    assert isinstance(config_base, MLPSpeculatorConfig)
    assert config_base.speculators_model_type == "mlp"
    assert config_base.hidden_size == 4096
    assert config_base.intermediate_size == 4096
    assert config_base.num_layers == 5
    assert config_base.tie_weights is True
    assert config_base.speculators_config.algorithm == "mlp"


@pytest.mark.smoke
def test_mlp_speculator_config_dict_marshalling(mlp_config_dict):
    original_config = MLPSpeculatorConfig.model_validate(mlp_config_dict)

    # Convert to dict with model_dump
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "mlp"
    assert config_dict["tie_weights"] is True
    assert config_dict["num_layers"] == 5

    # Load with from_dict on base class
    recreated_base = SpeculatorModelConfig.from_dict(config_dict)
    assert isinstance(recreated_base, MLPSpeculatorConfig)
    assert recreated_base.tie_weights is True
    assert recreated_base.num_layers == 5
    assert recreated_base.hidden_size == 4096

    # Load with from_dict on derived class (should work through inheritance)
    recreated_derived = MLPSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, MLPSpeculatorConfig)
    assert recreated_derived.tie_weights is True
    assert recreated_derived.num_layers == 5
    assert recreated_derived.hidden_size == 4096


@pytest.mark.smoke
def test_mlp_speculator_config_from_pretrained_local_marshalling(mlp_config_dict):
    original_config = MLPSpeculatorConfig.model_validate(mlp_config_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save with save_pretrained
        original_config.save_pretrained(temp_path)

        # Verify config.json was created
        config_file = temp_path / "config.json"
        assert config_file.exists()

        # Load with from_pretrained on base class
        loaded_base = SpeculatorModelConfig.from_pretrained(temp_path)
        assert isinstance(loaded_base, MLPSpeculatorConfig)
        assert loaded_base.speculators_model_type == "mlp"
        assert loaded_base.tie_weights is True
        assert loaded_base.num_layers == 5
        assert loaded_base.hidden_size == 4096

        # Load with from_pretrained on derived class
        loaded_derived = MLPSpeculatorConfig.from_pretrained(temp_path)
        assert isinstance(loaded_derived, MLPSpeculatorConfig)
        assert loaded_derived.speculators_model_type == "mlp"
        assert loaded_derived.tie_weights is True
        assert loaded_derived.num_layers == 5
        assert loaded_derived.hidden_size == 4096


@pytest.mark.smoke
def test_mlp_speculator_config_parameter_validation():
    """Test parameter validation for MLP-specific fields."""

    # Test valid dtype values
    valid_dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.int4]
    for dtype in valid_dtypes:
        config = MLPSpeculatorConfig(dtype=dtype)
        assert config.dtype == dtype

    # Test valid inputs
    valid_inputs = [
        ["input_embeddings"],
        ["hidden_states[-1]"],
        ["input_embeddings", "hidden_states[-1]"],
        ["custom_input"],
    ]
    for inputs in valid_inputs:
        config = MLPSpeculatorConfig(inputs=inputs)
        assert config.inputs == inputs

    # Test positive integer fields
    config = MLPSpeculatorConfig(
        hidden_size=1024,
        intermediate_size=2048,
        vocab_size=50000,
        num_layers=3,
    )
    assert config.hidden_size == 1024
    assert config.intermediate_size == 2048
    assert config.vocab_size == 50000
    assert config.num_layers == 3

    # Test boolean fields
    config = MLPSpeculatorConfig(
        inputs_hidden_states_normalized=True,
        tie_weights=False,
    )
    assert config.inputs_hidden_states_normalized is True
    assert config.tie_weights is False


@pytest.mark.smoke
def test_mlp_speculator_config_edge_cases():
    """Test edge cases for MLP configuration."""

    # Test minimum values
    config = MLPSpeculatorConfig(
        hidden_size=1,
        intermediate_size=1,
        vocab_size=1,
        num_layers=1,
    )
    assert config.hidden_size == 1
    assert config.intermediate_size == 1
    assert config.vocab_size == 1
    assert config.num_layers == 1

    # Test large values
    config = MLPSpeculatorConfig(
        hidden_size=65536,
        intermediate_size=131072,
        vocab_size=1000000,
        num_layers=20,
    )
    assert config.hidden_size == 65536
    assert config.intermediate_size == 131072
    assert config.vocab_size == 1000000
    assert config.num_layers == 20

    # Test zero intermediate_size (should use hidden_size)
    config = MLPSpeculatorConfig(
        hidden_size=2048,
        intermediate_size=0,
    )
    assert config.hidden_size == 2048
    assert config.intermediate_size == 0  # This will be handled in the model
