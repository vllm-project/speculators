"""
Unit tests for the dynamic tree proposal module in the Speculators library.
"""

import pytest
from pydantic import BaseModel, ValidationError

from speculators.config import TokenProposalConfig
from speculators.proposals import DynamicTreeTokenProposalConfig

# ===== DynamicTreeTokenProposalConfig Tests =====


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_initialization():
    config = DynamicTreeTokenProposalConfig()
    assert config.proposal_type == "dynamic_tree"
    assert config.speculative_tokens == 48
    assert config.expansion_factor == 10
    assert config.depth == 6


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_base_initialization():
    # create base instance to test initialization through TokenProposalConfig
    config = DynamicTreeTokenProposalConfig(
        speculative_tokens=36, expansion_factor=8, depth=4
    )
    config_dict = config.model_dump()

    # Validate the base class initialization
    config_base = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(config_base, DynamicTreeTokenProposalConfig)
    assert config_base.proposal_type == "dynamic_tree"
    assert config_base.speculative_tokens == 36
    assert config_base.expansion_factor == 8
    assert config_base.depth == 4


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_nested_initialization():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        dynamic_proposal: DynamicTreeTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        dynamic_proposals_list: list[DynamicTreeTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        dynamic_proposals_dict: dict[str, DynamicTreeTokenProposalConfig]

    parent = ParentModel(
        proposal=DynamicTreeTokenProposalConfig(speculative_tokens=24),
        dynamic_proposal=DynamicTreeTokenProposalConfig(speculative_tokens=12),
        proposals_list=[
            DynamicTreeTokenProposalConfig(speculative_tokens=6),
        ],
        dynamic_proposals_list=[
            DynamicTreeTokenProposalConfig(speculative_tokens=8),
        ],
        proposals_dict={
            "first": DynamicTreeTokenProposalConfig(speculative_tokens=10),
        },
        dynamic_proposals_dict={
            "second": DynamicTreeTokenProposalConfig(speculative_tokens=14),
        },
    )

    # Validate parent model properties
    assert isinstance(parent.proposal, DynamicTreeTokenProposalConfig)
    assert parent.proposal.speculative_tokens == 24
    assert isinstance(parent.dynamic_proposal, DynamicTreeTokenProposalConfig)
    assert parent.dynamic_proposal.speculative_tokens == 12
    assert len(parent.proposals_list) == 1
    assert isinstance(parent.proposals_list[0], DynamicTreeTokenProposalConfig)
    assert parent.proposals_list[0].speculative_tokens == 6
    assert len(parent.dynamic_proposals_list) == 1
    assert isinstance(parent.dynamic_proposals_list[0], DynamicTreeTokenProposalConfig)
    assert parent.dynamic_proposals_list[0].speculative_tokens == 8
    assert len(parent.proposals_dict) == 1
    assert isinstance(parent.proposals_dict["first"], DynamicTreeTokenProposalConfig)
    assert parent.proposals_dict["first"].speculative_tokens == 10
    assert len(parent.dynamic_proposals_dict) == 1
    assert isinstance(
        parent.dynamic_proposals_dict["second"], DynamicTreeTokenProposalConfig
    )
    assert parent.dynamic_proposals_dict["second"].speculative_tokens == 14


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_invalid_initialization():
    # Test with invalid proposal_type
    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(proposal_type="invalid_type")  # type: ignore
    assert "proposal_type" in str(exc_info.value)

    # Test with invalid speculative_tokens (negative value)
    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(speculative_tokens=0)
    assert "speculative_tokens" in str(exc_info.value)

    # Test with invalid expansion_factor (negative value)
    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(expansion_factor=0)
    assert "expansion_factor" in str(exc_info.value)

    # Test with invalid depth (less than 1)
    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(depth=0)
    assert "depth" in str(exc_info.value)

    # Test with non-integer values
    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(speculative_tokens="invalid")  # type: ignore
    assert "speculative_tokens" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(expansion_factor="invalid")  # type: ignore
    assert "expansion_factor" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        DynamicTreeTokenProposalConfig(depth="invalid")  # type: ignore
    assert "depth" in str(exc_info.value)


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_marshalling():
    # Create original config with custom values
    original_config = DynamicTreeTokenProposalConfig(
        speculative_tokens=36, expansion_factor=8, depth=4
    )

    # Convert to dict
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["proposal_type"] == "dynamic_tree"
    assert config_dict["speculative_tokens"] == 36
    assert config_dict["expansion_factor"] == 8
    assert config_dict["depth"] == 4

    # Recreate from dict using model_validate on base class
    recreated_config = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, DynamicTreeTokenProposalConfig)
    assert recreated_config.proposal_type == original_config.proposal_type
    assert recreated_config.speculative_tokens == original_config.speculative_tokens
    assert recreated_config.expansion_factor == original_config.expansion_factor
    assert recreated_config.depth == original_config.depth

    # Recreate from dict using model_validate on derived class
    recreated_config = DynamicTreeTokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, DynamicTreeTokenProposalConfig)
    assert recreated_config.proposal_type == original_config.proposal_type
    assert recreated_config.speculative_tokens == original_config.speculative_tokens
    assert recreated_config.expansion_factor == original_config.expansion_factor
    assert recreated_config.depth == original_config.depth


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_parent_marshalling():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        dynamic_proposal: DynamicTreeTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        dynamic_proposals_list: list[DynamicTreeTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        dynamic_proposals_dict: dict[str, DynamicTreeTokenProposalConfig]

    # Create original parent model
    original_parent = ParentModel(
        proposal=DynamicTreeTokenProposalConfig(speculative_tokens=24),
        dynamic_proposal=DynamicTreeTokenProposalConfig(speculative_tokens=12),
        proposals_list=[
            DynamicTreeTokenProposalConfig(speculative_tokens=6),
        ],
        dynamic_proposals_list=[
            DynamicTreeTokenProposalConfig(speculative_tokens=8),
        ],
        proposals_dict={
            "first": DynamicTreeTokenProposalConfig(speculative_tokens=10),
        },
        dynamic_proposals_dict={
            "second": DynamicTreeTokenProposalConfig(speculative_tokens=14),
        },
    )

    # Convert to dict
    parent_dict = original_parent.model_dump()
    parent = ParentModel.model_validate(parent_dict)

    # Validate parent model and dict properties are correct types and match original
    assert isinstance(parent_dict, dict)
    assert isinstance(parent, ParentModel)
    assert isinstance(parent_dict["proposal"], dict)
    assert isinstance(parent.proposal, DynamicTreeTokenProposalConfig)
    assert (
        24
        == parent_dict["proposal"]["speculative_tokens"]
        == parent.proposal.speculative_tokens
    )
    assert isinstance(parent_dict["dynamic_proposal"], dict)
    assert isinstance(parent.dynamic_proposal, DynamicTreeTokenProposalConfig)
    assert (
        12
        == parent_dict["dynamic_proposal"]["speculative_tokens"]
        == parent.dynamic_proposal.speculative_tokens
    )
    assert isinstance(parent_dict["proposals_list"], list)
    assert len(parent_dict["proposals_list"]) == 1
    assert isinstance(parent.proposals_list, list)
    assert len(parent.proposals_list) == 1
    assert isinstance(parent_dict["proposals_list"][0], dict)
    assert isinstance(parent.proposals_list[0], DynamicTreeTokenProposalConfig)
    assert (
        6
        == parent_dict["proposals_list"][0]["speculative_tokens"]
        == parent.proposals_list[0].speculative_tokens
    )
    assert isinstance(parent_dict["dynamic_proposals_list"], list)
    assert len(parent_dict["dynamic_proposals_list"]) == 1
    assert isinstance(parent.dynamic_proposals_list, list)
    assert len(parent.dynamic_proposals_list) == 1
    assert isinstance(parent_dict["dynamic_proposals_list"][0], dict)
    assert isinstance(parent.dynamic_proposals_list[0], DynamicTreeTokenProposalConfig)
    assert (
        8
        == parent_dict["dynamic_proposals_list"][0]["speculative_tokens"]
        == parent.dynamic_proposals_list[0].speculative_tokens
    )
    assert isinstance(parent_dict["proposals_dict"], dict)
    assert len(parent_dict["proposals_dict"]) == 1
    assert isinstance(parent.proposals_dict, dict)
    assert len(parent.proposals_dict) == 1
    assert isinstance(parent_dict["proposals_dict"]["first"], dict)
    assert isinstance(parent.proposals_dict["first"], DynamicTreeTokenProposalConfig)
    assert (
        10
        == parent_dict["proposals_dict"]["first"]["speculative_tokens"]
        == parent.proposals_dict["first"].speculative_tokens
    )
    assert isinstance(parent_dict["dynamic_proposals_dict"], dict)
    assert len(parent_dict["dynamic_proposals_dict"]) == 1
    assert isinstance(parent.dynamic_proposals_dict, dict)
    assert len(parent.dynamic_proposals_dict) == 1
    assert isinstance(parent_dict["dynamic_proposals_dict"]["second"], dict)
    assert isinstance(
        parent.dynamic_proposals_dict["second"], DynamicTreeTokenProposalConfig
    )
    assert (
        14
        == parent_dict["dynamic_proposals_dict"]["second"]["speculative_tokens"]
        == parent.dynamic_proposals_dict["second"].speculative_tokens
    )


@pytest.mark.smoke
def test_dynamic_tree_token_proposal_config_compiled_loading():
    proposal_dict = {
        "proposal_type": "dynamic_tree",
        "speculative_tokens": 48,
        "expansion_factor": 10,
        "depth": 6,
    }
    proposal = TokenProposalConfig.model_validate(proposal_dict)
    assert isinstance(proposal, DynamicTreeTokenProposalConfig)
    assert proposal.proposal_type == "dynamic_tree"
    assert proposal.speculative_tokens == 48
    assert proposal.expansion_factor == 10
    assert proposal.depth == 6
