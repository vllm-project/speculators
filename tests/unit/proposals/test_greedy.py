"""
Unit tests for the greedy proposal module in the Speculators library.
"""

import pytest
from pydantic import BaseModel, ValidationError

from speculators.config import TokenProposalConfig
from speculators.proposals import GreedyTokenProposalConfig

# ===== GreedyTokenProposalConfig Tests =====


@pytest.mark.smoke
def test_greedy_token_proposal_config_initialization():
    config = GreedyTokenProposalConfig()
    assert config.proposal_type == "greedy"
    assert config.speculative_tokens == 5
    assert config.verifier_accept_k == 1
    assert config.accept_tolerance == 0.0


@pytest.mark.smoke
def test_greedy_token_proposal_config_base_initialization():
    # create base instance to test initialization through TokenProposalConfig
    config = GreedyTokenProposalConfig(
        speculative_tokens=10, verifier_accept_k=3, accept_tolerance=1.5
    )
    config_dict = config.model_dump()

    # Validate the base class initialization
    config_base = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(config_base, GreedyTokenProposalConfig)
    assert config_base.proposal_type == "greedy"
    assert config_base.speculative_tokens == 10
    assert config_base.verifier_accept_k == 3
    assert config_base.accept_tolerance == 1.5


@pytest.mark.smoke
def test_greedy_token_proposal_config_nested_initialization():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        greedy_proposal: GreedyTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        greedy_proposals_list: list[GreedyTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        greedy_proposals_dict: dict[str, GreedyTokenProposalConfig]

    parent = ParentModel(
        proposal=GreedyTokenProposalConfig(speculative_tokens=10),
        greedy_proposal=GreedyTokenProposalConfig(verifier_accept_k=3),
        proposals_list=[
            GreedyTokenProposalConfig(speculative_tokens=8),
        ],
        greedy_proposals_list=[
            GreedyTokenProposalConfig(speculative_tokens=6),
        ],
        proposals_dict={
            "first": GreedyTokenProposalConfig(speculative_tokens=12),
        },
        greedy_proposals_dict={
            "second": GreedyTokenProposalConfig(speculative_tokens=15),
        },
    )

    # Validate parent model properties
    assert isinstance(parent.proposal, GreedyTokenProposalConfig)
    assert parent.proposal.speculative_tokens == 10
    assert isinstance(parent.greedy_proposal, GreedyTokenProposalConfig)
    assert parent.greedy_proposal.verifier_accept_k == 3
    assert len(parent.proposals_list) == 1
    assert isinstance(parent.proposals_list[0], GreedyTokenProposalConfig)
    assert parent.proposals_list[0].speculative_tokens == 8
    assert len(parent.greedy_proposals_list) == 1
    assert isinstance(parent.greedy_proposals_list[0], GreedyTokenProposalConfig)
    assert parent.greedy_proposals_list[0].speculative_tokens == 6
    assert len(parent.proposals_dict) == 1
    assert isinstance(parent.proposals_dict["first"], GreedyTokenProposalConfig)
    assert parent.proposals_dict["first"].speculative_tokens == 12
    assert len(parent.greedy_proposals_dict) == 1
    assert isinstance(parent.greedy_proposals_dict["second"], GreedyTokenProposalConfig)
    assert parent.greedy_proposals_dict["second"].speculative_tokens == 15


@pytest.mark.smoke
def test_greedy_token_proposal_config_invalid_initialization():
    # Test with invalid proposal_type
    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(proposal_type="invalid")  # type: ignore[arg-type]
    assert "proposal_type" in str(exc_info.value)

    # Test with invalid speculative_tokens (negative value)
    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(speculative_tokens=-1)
    assert "speculative_tokens" in str(exc_info.value)

    # Test with invalid verifier_accept_k (negative value)
    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(verifier_accept_k=-1)
    assert "verifier_accept_k" in str(exc_info.value)

    # Test with invalid accept_tolerance (negative value)
    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(accept_tolerance=-1.0)
    assert "accept_tolerance" in str(exc_info.value)

    # Test with non-integer values
    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(speculative_tokens=5.5)  # type: ignore[arg-type]
    assert "speculative_tokens" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(verifier_accept_k=3.5)  # type: ignore[arg-type]
    assert "verifier_accept_k" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        GreedyTokenProposalConfig(accept_tolerance="high")  # type: ignore[arg-type]
    assert "accept_tolerance" in str(exc_info.value)


@pytest.mark.smoke
def test_greedy_token_proposal_config_marshalling():
    # Create original config with custom values
    original_config = GreedyTokenProposalConfig(
        speculative_tokens=10, verifier_accept_k=3, accept_tolerance=1.5
    )

    # Convert to dict
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["proposal_type"] == "greedy"
    assert config_dict["speculative_tokens"] == 10
    assert config_dict["verifier_accept_k"] == 3
    assert config_dict["accept_tolerance"] == 1.5

    # Recreate from dict using model_validate on base class
    recreated_config = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, GreedyTokenProposalConfig)
    assert recreated_config.proposal_type == original_config.proposal_type
    assert recreated_config.speculative_tokens == original_config.speculative_tokens
    assert recreated_config.verifier_accept_k == original_config.verifier_accept_k
    assert recreated_config.accept_tolerance == original_config.accept_tolerance

    # Recreate from dict using model_validate on derived class
    recreated_config = GreedyTokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, GreedyTokenProposalConfig)
    assert recreated_config.proposal_type == original_config.proposal_type
    assert recreated_config.speculative_tokens == original_config.speculative_tokens
    assert recreated_config.verifier_accept_k == original_config.verifier_accept_k
    assert recreated_config.accept_tolerance == original_config.accept_tolerance


@pytest.mark.smoke
def test_greedy_token_proposal_config_parent_marshalling():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        greedy_proposal: GreedyTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        greedy_proposals_list: list[GreedyTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        greedy_proposals_dict: dict[str, GreedyTokenProposalConfig]

    # Create original parent model
    original_parent = ParentModel(
        proposal=GreedyTokenProposalConfig(speculative_tokens=10),
        greedy_proposal=GreedyTokenProposalConfig(verifier_accept_k=3),
        proposals_list=[
            GreedyTokenProposalConfig(speculative_tokens=8),
        ],
        greedy_proposals_list=[
            GreedyTokenProposalConfig(speculative_tokens=6),
        ],
        proposals_dict={
            "first": GreedyTokenProposalConfig(speculative_tokens=12),
        },
        greedy_proposals_dict={
            "second": GreedyTokenProposalConfig(speculative_tokens=15),
        },
    )

    # Convert to dict
    parent_dict = original_parent.model_dump()
    parent = ParentModel.model_validate(parent_dict)

    # Validate parent model and dict properties are correct types and match original
    assert isinstance(parent_dict, dict)
    assert isinstance(parent, ParentModel)
    assert isinstance(parent_dict["proposal"], dict)
    assert isinstance(parent.proposal, GreedyTokenProposalConfig)
    assert (
        10
        == parent_dict["proposal"]["speculative_tokens"]
        == parent.proposal.speculative_tokens
    )
    assert isinstance(parent_dict["greedy_proposal"], dict)
    assert isinstance(parent.greedy_proposal, GreedyTokenProposalConfig)
    assert (
        3
        == parent_dict["greedy_proposal"]["verifier_accept_k"]
        == parent.greedy_proposal.verifier_accept_k
    )
    assert isinstance(parent_dict["proposals_list"], list)
    assert len(parent_dict["proposals_list"]) == 1
    assert isinstance(parent.proposals_list, list)
    assert len(parent.proposals_list) == 1
    assert isinstance(parent_dict["proposals_list"][0], dict)
    assert isinstance(parent.proposals_list[0], GreedyTokenProposalConfig)
    assert (
        8
        == parent_dict["proposals_list"][0]["speculative_tokens"]
        == parent.proposals_list[0].speculative_tokens
    )
    assert isinstance(parent_dict["greedy_proposals_list"], list)
    assert len(parent_dict["greedy_proposals_list"]) == 1
    assert isinstance(parent.greedy_proposals_list, list)
    assert len(parent.greedy_proposals_list) == 1
    assert isinstance(parent_dict["greedy_proposals_list"][0], dict)
    assert isinstance(parent.greedy_proposals_list[0], GreedyTokenProposalConfig)
    assert (
        6
        == parent_dict["greedy_proposals_list"][0]["speculative_tokens"]
        == parent.greedy_proposals_list[0].speculative_tokens
    )
    assert isinstance(parent_dict["proposals_dict"], dict)
    assert len(parent_dict["proposals_dict"]) == 1
    assert isinstance(parent.proposals_dict, dict)
    assert isinstance(parent_dict["proposals_dict"]["first"], dict)
    assert isinstance(parent.proposals_dict["first"], GreedyTokenProposalConfig)
    assert (
        12
        == parent_dict["proposals_dict"]["first"]["speculative_tokens"]
        == parent.proposals_dict["first"].speculative_tokens
    )
    assert isinstance(parent_dict["greedy_proposals_dict"], dict)
    assert len(parent_dict["greedy_proposals_dict"]) == 1
    assert isinstance(parent.greedy_proposals_dict, dict)
    assert isinstance(parent_dict["greedy_proposals_dict"]["second"], dict)
    assert isinstance(parent.greedy_proposals_dict["second"], GreedyTokenProposalConfig)
    assert (
        15
        == parent_dict["greedy_proposals_dict"]["second"]["speculative_tokens"]
        == parent.greedy_proposals_dict["second"].speculative_tokens
    )


@pytest.mark.smoke
def test_greedy_token_proposal_compiled_loading():
    proposal_dict = {
        "proposal_type": "greedy",
        "speculative_tokens": 10,
        "verifier_accept_k": 3,
        "accept_tolerance": 1.5,
    }
    proposal = TokenProposalConfig.model_validate(proposal_dict)
    assert isinstance(proposal, GreedyTokenProposalConfig)
    assert proposal.proposal_type == "greedy"
    assert proposal.speculative_tokens == 10
    assert proposal.verifier_accept_k == 3
    assert proposal.accept_tolerance == 1.5
