"""
Unit tests for the eagle model module in the Speculators library.
"""

import pytest
from pydantic import BaseModel, ValidationError

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models import IndependentSpeculatorConfig
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
        algorithm="independent",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


@pytest.fixture
def independent_config_dict():
    return {
        "speculators_model_type": "independent",
        "speculators_config": {
            "algorithm": "independent",
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


# ===== EagleSpeculatorConfig Tests =====


def test_indepentent_speculator_from_pretrained():
    config = IndependentSpeculatorConfig.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct"
    )
    assert config.model_type == "llama"
    assert config.speculators_model_type == "independent"
    assert config.speculators_config is None


@pytest.mark.smoke
def test_independent_speculator_config_initialization():
    """Test default initialization of IndependentSpeculatorConfig."""
    config = IndependentSpeculatorConfig()

    # Verify Independent-specific defaults
    assert config.speculators_model_type == "independent"

    # Verify base class defaults
    assert config.model_type == "speculator_model"
    assert config.speculators_config is None


@pytest.mark.smoke
def test_independent_speculator_config_custom_initialization(sample_speculators_config):
    """Test custom initialization of IndependentSpeculatorConfig."""
    config = IndependentSpeculatorConfig(speculators_config=sample_speculators_config)

    # Verify custom values
    assert config.speculators_model_type == "independent"
    assert config.speculators_config == sample_speculators_config


@pytest.mark.smoke
def test_independent_speculator_config_base_initialization(sample_speculators_config):
    # Create IndependentSpeculatorConfig with custom values
    original_config = IndependentSpeculatorConfig(
        speculators_config=sample_speculators_config,
    )

    # Convert to dict and validate through base class
    config_dict = original_config.model_dump()
    recreated_config = SpeculatorModelConfig.model_validate(config_dict)

    # Verify type and values preservation
    assert isinstance(recreated_config, IndependentSpeculatorConfig)
    assert recreated_config.speculators_model_type == "independent"
    assert recreated_config.speculators_config == sample_speculators_config


@pytest.mark.regression
def test_independent_speculator_config_nested_initialization():
    class ParentModel(BaseModel):
        single_config: IndependentSpeculatorConfig
        config_list: list[IndependentSpeculatorConfig]
        config_dict: dict[str, IndependentSpeculatorConfig]

    parent = ParentModel(
        single_config=IndependentSpeculatorConfig(),
        config_list=[
            IndependentSpeculatorConfig(),
            IndependentSpeculatorConfig(),
        ],
        config_dict={
            "draft1": IndependentSpeculatorConfig(),
            "draft2": IndependentSpeculatorConfig(),
        },
    )

    # Verify single config
    assert isinstance(parent.single_config, IndependentSpeculatorConfig)

    # Verify config list
    assert len(parent.config_list) == 2
    assert all(isinstance(c, IndependentSpeculatorConfig) for c in parent.config_list)

    # Verify config dict
    assert len(parent.config_dict) == 2
    assert all(
        isinstance(c, IndependentSpeculatorConfig) for c in parent.config_dict.values()
    )


@pytest.mark.smoke
def test_independent_speculator_config_invalid_initialization():
    # Test invalid speculators_model_type
    with pytest.raises(ValidationError) as exc_info:
        IndependentSpeculatorConfig(speculators_model_type="invalid")  # type: ignore[arg-type]
    assert "speculators_model_type" in str(exc_info.value)


@pytest.mark.smoke
def test_independent_speculator_config_auto_registry():
    registered_classes = SpeculatorModelConfig.registered_classes()
    class_names = [cls.__name__ for cls in registered_classes]

    # Verify IndependentSpeculatorConfig is registered
    assert "IndependentSpeculatorConfig" in class_names

    # Verify registry key mapping
    assert SpeculatorModelConfig.registry is not None
    assert "independent" in SpeculatorModelConfig.registry
    assert SpeculatorModelConfig.registry["independent"] == IndependentSpeculatorConfig


@pytest.mark.smoke
def test_independent_speculator_config_marshalling(sample_speculators_config):
    original_config = IndependentSpeculatorConfig(
        speculators_config=sample_speculators_config,
    )

    # Test model_dump()
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "independent"
    assert config_dict["speculators_config"] == sample_speculators_config.model_dump()

    # Test model_validate() on base class
    recreated_base = SpeculatorModelConfig.model_validate(config_dict)
    assert isinstance(recreated_base, IndependentSpeculatorConfig)
    assert recreated_base.speculators_config == sample_speculators_config

    # Test model_validate() on derived class
    recreated_derived = IndependentSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, IndependentSpeculatorConfig)
    assert recreated_derived.speculators_config == sample_speculators_config
