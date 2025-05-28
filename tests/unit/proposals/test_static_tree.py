"""
Unit tests for the static tree proposal module in the Speculators library.
"""

import pytest
from pydantic import BaseModel, ValidationError

from speculators.config import TokenProposalConfig
from speculators.proposals import StaticTreeTokenProposalConfig

# ===== StaticTreeTokenProposalConfig Tests =====


@pytest.mark.smoke
def test_static_tree_token_proposal_config_initialization():
    config = StaticTreeTokenProposalConfig()
    assert config.proposal_type == "static_tree"
    assert config.initial_branching_factor == 4
    assert config.branching_factor == 2
    assert config.depth == 5


@pytest.mark.smoke
def test_static_tree_token_proposal_config_base_initialization():
    # create base instance to test initialization through TokenProposalConfig
    config = StaticTreeTokenProposalConfig(
        initial_branching_factor=6, branching_factor=3, depth=4
    )
    config_dict = config.model_dump()

    # Validate the base class initialization
    config_base = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(config_base, StaticTreeTokenProposalConfig)
    assert config_base.proposal_type == "static_tree"
    assert config_base.initial_branching_factor == 6
    assert config_base.branching_factor == 3
    assert config_base.depth == 4


@pytest.mark.smoke
def test_static_tree_token_proposal_config_nested_initialization():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        static_proposal: StaticTreeTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        static_proposals_list: list[StaticTreeTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        static_proposals_dict: dict[str, StaticTreeTokenProposalConfig]

    parent = ParentModel(
        proposal=StaticTreeTokenProposalConfig(initial_branching_factor=8),
        static_proposal=StaticTreeTokenProposalConfig(branching_factor=4),
        proposals_list=[
            StaticTreeTokenProposalConfig(depth=6),
        ],
        static_proposals_list=[
            StaticTreeTokenProposalConfig(initial_branching_factor=5),
        ],
        proposals_dict={
            "first": StaticTreeTokenProposalConfig(branching_factor=3),
        },
        static_proposals_dict={
            "second": StaticTreeTokenProposalConfig(depth=7),
        },
    )

    # Validate parent model properties
    assert isinstance(parent.proposal, StaticTreeTokenProposalConfig)
    assert parent.proposal.initial_branching_factor == 8
    assert isinstance(parent.static_proposal, StaticTreeTokenProposalConfig)
    assert parent.static_proposal.branching_factor == 4
    assert len(parent.proposals_list) == 1
    assert isinstance(parent.proposals_list[0], StaticTreeTokenProposalConfig)
    assert parent.proposals_list[0].depth == 6
    assert len(parent.static_proposals_list) == 1
    assert isinstance(parent.static_proposals_list[0], StaticTreeTokenProposalConfig)
    assert parent.static_proposals_list[0].initial_branching_factor == 5
    assert len(parent.proposals_dict) == 1
    assert isinstance(parent.proposals_dict["first"], StaticTreeTokenProposalConfig)
    assert parent.proposals_dict["first"].branching_factor == 3
    assert len(parent.static_proposals_dict) == 1
    assert isinstance(
        parent.static_proposals_dict["second"], StaticTreeTokenProposalConfig
    )
    assert parent.static_proposals_dict["second"].depth == 7


@pytest.mark.smoke
def test_static_tree_token_proposal_config_invalid_initialization():
    # Test with invalid proposal_type
    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(proposal_type="invalid_type")  # type: ignore
    assert "proposal_type" in str(exc_info.value)

    # Test with invalid initial_branching_factor (less than 1)
    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(initial_branching_factor=0)
    assert "initial_branching_factor" in str(exc_info.value)

    # Test with invalid branching_factor (less than 1)
    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(branching_factor=0)
    assert "branching_factor" in str(exc_info.value)

    # Test with invalid depth (less than 1)
    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(depth=0)
    assert "depth" in str(exc_info.value)

    # Test with non-integer values
    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(initial_branching_factor="invalid")  # type: ignore
    assert "initial_branching_factor" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(branching_factor="invalid")  # type: ignore
    assert "branching_factor" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        StaticTreeTokenProposalConfig(depth="invalid")  # type: ignore
    assert "depth" in str(exc_info.value)


@pytest.mark.smoke
def test_static_tree_token_proposal_config_marshalling():
    # Create original config with custom values
    original_config = StaticTreeTokenProposalConfig(
        initial_branching_factor=6, branching_factor=3, depth=4
    )

    # Convert to dict
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["proposal_type"] == "static_tree"
    assert config_dict["initial_branching_factor"] == 6
    assert config_dict["branching_factor"] == 3
    assert config_dict["depth"] == 4

    # Recreate from dict using model_validate on base class
    recreated_config = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, StaticTreeTokenProposalConfig)
    assert recreated_config.proposal_type == original_config.proposal_type
    assert (
        recreated_config.initial_branching_factor
        == original_config.initial_branching_factor
    )
    assert recreated_config.branching_factor == original_config.branching_factor
    assert recreated_config.depth == original_config.depth

    # Recreate from dict using model_validate on derived class
    recreated_config = StaticTreeTokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, StaticTreeTokenProposalConfig)
    assert recreated_config.proposal_type == original_config.proposal_type
    assert (
        recreated_config.initial_branching_factor
        == original_config.initial_branching_factor
    )
    assert recreated_config.branching_factor == original_config.branching_factor
    assert recreated_config.depth == original_config.depth


@pytest.mark.smoke
def test_static_tree_token_proposal_config_parent_marshalling():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        static_proposal: StaticTreeTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        static_proposals_list: list[StaticTreeTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        static_proposals_dict: dict[str, StaticTreeTokenProposalConfig]

    # Create original parent model
    original_parent = ParentModel(
        proposal=StaticTreeTokenProposalConfig(initial_branching_factor=8),
        static_proposal=StaticTreeTokenProposalConfig(branching_factor=4),
        proposals_list=[
            StaticTreeTokenProposalConfig(depth=6),
        ],
        static_proposals_list=[
            StaticTreeTokenProposalConfig(initial_branching_factor=5),
        ],
        proposals_dict={
            "first": StaticTreeTokenProposalConfig(branching_factor=3),
        },
        static_proposals_dict={
            "second": StaticTreeTokenProposalConfig(depth=7),
        },
    )

    # Convert to dict
    parent_dict = original_parent.model_dump()
    parent = ParentModel.model_validate(parent_dict)

    # Validate parent model and dict properties are correct types and match original
    assert isinstance(parent_dict, dict)
    assert isinstance(parent, ParentModel)
    assert isinstance(parent_dict["proposal"], dict)
    assert isinstance(parent.proposal, StaticTreeTokenProposalConfig)
    assert (
        8
        == parent_dict["proposal"]["initial_branching_factor"]
        == parent.proposal.initial_branching_factor
    )
    assert isinstance(parent_dict["static_proposal"], dict)
    assert isinstance(parent.static_proposal, StaticTreeTokenProposalConfig)
    assert (
        4
        == parent_dict["static_proposal"]["branching_factor"]
        == parent.static_proposal.branching_factor
    )
    assert isinstance(parent_dict["proposals_list"], list)
    assert len(parent_dict["proposals_list"]) == 1
    assert isinstance(parent.proposals_list, list)
    assert len(parent.proposals_list) == 1
    assert isinstance(parent_dict["proposals_list"][0], dict)
    assert isinstance(parent.proposals_list[0], StaticTreeTokenProposalConfig)
    assert (
        6 == parent_dict["proposals_list"][0]["depth"] == parent.proposals_list[0].depth
    )
    assert isinstance(parent_dict["static_proposals_list"], list)
    assert len(parent_dict["static_proposals_list"]) == 1
    assert isinstance(parent.static_proposals_list, list)
    assert len(parent.static_proposals_list) == 1
    assert isinstance(parent_dict["static_proposals_list"][0], dict)
    assert isinstance(parent.static_proposals_list[0], StaticTreeTokenProposalConfig)
    assert (
        5
        == parent_dict["static_proposals_list"][0]["initial_branching_factor"]
        == parent.static_proposals_list[0].initial_branching_factor
    )


@pytest.mark.smoke
def test_static_tree_token_proposal_config_compiled_loading():
    proposal_dict = {
        "proposal_type": "static_tree",
        "initial_branching_factor": 4,
        "branching_factor": 2,
        "depth": 5,
    }
    proposal = TokenProposalConfig.model_validate(proposal_dict)
    assert isinstance(proposal, StaticTreeTokenProposalConfig)
    assert proposal.proposal_type == "static_tree"
    assert proposal.initial_branching_factor == 4
    assert proposal.branching_factor == 2
    assert proposal.depth == 5
