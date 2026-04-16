"""Integration tests for the algorithm registry system in the Speculators library.

Tests the full register -> resolve -> instantiate pipeline to verify that
the registry infrastructure works end-to-end for algorithm discovery.
"""

from typing import Literal

import pytest

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    TokenProposalConfig,
    VerifierConfig,
    reload_schemas,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.registry import ClassRegistryMixin

# ===== Test-Specific Registry Subclasses =====


@TokenProposalConfig.register("integ_test_proposal")
class IntegTestProposalConfig(TokenProposalConfig):
    proposal_type: Literal["integ_test_proposal"] = "integ_test_proposal"
    test_beam_width: int = 4


@SpeculatorModelConfig.register("integ_test_algo")
class IntegTestAlgoConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["integ_test_algo"] = "integ_test_algo"
    custom_block_size: int = 8
    custom_num_heads: int = 4


reload_schemas()


# ===== Fixtures =====


@pytest.fixture
def verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier-model",
        architectures=["TestForCausalLM"],
    )


@pytest.fixture
def greedy_proposal():
    return GreedyTokenProposalConfig(speculative_tokens=5)


@pytest.fixture
def speculators_config(greedy_proposal, verifier_config):
    return SpeculatorsConfig(
        algorithm="integ_test_algo",
        proposal_methods=[greedy_proposal],
        default_proposal_method="greedy",
        verifier=verifier_config,
    )


@pytest.fixture
def full_config(speculators_config):
    return IntegTestAlgoConfig(
        speculators_model_type="integ_test_algo",
        speculators_config=speculators_config,
        custom_block_size=16,
        custom_num_heads=8,
    )


# ===== Registration Tests =====


@pytest.mark.smoke
def test_registry_contains_builtin_algorithms():
    """Built-in algorithms (eagle3) should be discoverable via the registry."""
    assert SpeculatorModelConfig.registry is not None
    assert "eagle3" in SpeculatorModelConfig.registry


@pytest.mark.smoke
def test_registry_contains_test_algorithm():
    """Test-registered algorithm should be discoverable."""
    assert SpeculatorModelConfig.registry is not None
    assert "integ_test_algo" in SpeculatorModelConfig.registry
    assert SpeculatorModelConfig.registry["integ_test_algo"] is IntegTestAlgoConfig


@pytest.mark.sanity
def test_registered_classes_returns_all():
    """registered_classes() includes both built-in and test-registered types."""
    classes = SpeculatorModelConfig.registered_classes()
    class_names = [cls.__name__ for cls in classes]
    assert "Eagle3SpeculatorConfig" in class_names
    assert "IntegTestAlgoConfig" in class_names


@pytest.mark.sanity
def test_proposal_registry_contains_builtin_and_test():
    """TokenProposalConfig registry includes greedy and test proposal."""
    classes = TokenProposalConfig.registered_classes()
    class_names = [cls.__name__ for cls in classes]
    assert "GreedyTokenProposalConfig" in class_names
    assert "IntegTestProposalConfig" in class_names


@pytest.mark.regression
def test_duplicate_registration_raises():
    """Attempting to register the same name twice should raise ValueError."""

    class FreshRegistry(ClassRegistryMixin):
        pass

    @FreshRegistry.register("unique_algo")
    class FirstAlgo:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @FreshRegistry.register("unique_algo")
        class SecondAlgo:
            pass


# ===== Resolve Tests =====


@pytest.mark.smoke
def test_resolve_config_via_model_validate(full_config):
    """model_validate dispatches to the correct subclass via discriminator."""
    config_dict = full_config.model_dump()
    resolved = SpeculatorModelConfig.model_validate(config_dict)

    assert isinstance(resolved, IntegTestAlgoConfig)
    assert resolved.speculators_model_type == "integ_test_algo"
    assert resolved.custom_block_size == 16
    assert resolved.custom_num_heads == 8


@pytest.mark.smoke
def test_resolve_config_via_from_dict(full_config):
    """from_dict should dispatch to the correct subclass."""
    config_dict = full_config.to_dict()
    resolved = SpeculatorModelConfig.from_dict(config_dict)

    assert isinstance(resolved, IntegTestAlgoConfig)
    assert resolved.speculators_model_type == "integ_test_algo"


@pytest.mark.sanity
def test_resolve_greedy_proposal_via_model_validate():
    """TokenProposalConfig should resolve greedy proposal from dict."""
    proposal = GreedyTokenProposalConfig(speculative_tokens=3)
    config_dict = proposal.model_dump()
    resolved = TokenProposalConfig.model_validate(config_dict)

    assert isinstance(resolved, GreedyTokenProposalConfig)
    assert resolved.speculative_tokens == 3


# ===== Instantiate Tests =====


@pytest.mark.sanity
def test_instantiate_full_config_from_parts(verifier_config):
    """Build a complete SpeculatorModelConfig from individual components."""
    proposal = GreedyTokenProposalConfig(speculative_tokens=7)
    spec_config = SpeculatorsConfig(
        algorithm="integ_test_algo",
        proposal_methods=[proposal],
        default_proposal_method="greedy",
        verifier=verifier_config,
    )
    model_config = IntegTestAlgoConfig(
        speculators_model_type="integ_test_algo",
        speculators_config=spec_config,
        custom_block_size=32,
    )

    assert model_config.speculators_model_type == "integ_test_algo"
    assert model_config.speculators_config.algorithm == "integ_test_algo"
    assert (
        model_config.speculators_config.verifier.name_or_path == "test/verifier-model"
    )
    assert model_config.speculators_config.proposal_methods[0].speculative_tokens == 7
    assert model_config.custom_block_size == 32


@pytest.mark.regression
def test_end_to_end_register_resolve_instantiate():
    """Full pipeline: register a new type, build, serialize, deserialize."""

    @SpeculatorModelConfig.register("e2e_test_model")
    class E2ETestModelConfig(SpeculatorModelConfig):
        speculators_model_type: Literal["e2e_test_model"] = "e2e_test_model"
        hidden_dim: int = 512

    reload_schemas()

    # Build using built-in GreedyTokenProposalConfig for round-trip stability
    proposal = GreedyTokenProposalConfig(speculative_tokens=10)
    verifier = VerifierConfig(
        name_or_path="test/e2e-verifier", architectures=["E2EModel"]
    )
    spec_config = SpeculatorsConfig(
        algorithm="e2e_test_model",
        proposal_methods=[proposal],
        default_proposal_method="greedy",
        verifier=verifier,
    )
    config = E2ETestModelConfig(
        speculators_config=spec_config,
        hidden_dim=1024,
    )

    # Serialize
    config_dict = config.model_dump()

    # Deserialize (should dispatch correctly)
    restored = SpeculatorModelConfig.model_validate(config_dict)

    assert isinstance(restored, E2ETestModelConfig)
    assert restored.hidden_dim == 1024
    assert restored.speculators_config.algorithm == "e2e_test_model"
    assert restored.speculators_config.proposal_methods[0].speculative_tokens == 10
    assert restored.speculators_config.verifier.name_or_path == "test/e2e-verifier"
