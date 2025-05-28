"""
Unit tests for the sampling proposal module in the Speculators library.
"""

import pytest
from pydantic import BaseModel, ValidationError

from speculators.config import TokenProposalConfig
from speculators.proposals import SamplingTokenProposalConfig

# ===== SamplingTokenProposalConfig Tests =====


@pytest.mark.smoke
def test_sampling_token_proposal_config_initialization():
    config = SamplingTokenProposalConfig()
    assert isinstance(config, SamplingTokenProposalConfig)
    assert config.proposal_type == "sampling"
    assert config.speculative_tokens == 5
    assert config.accept_lenience == 1.0


@pytest.mark.smoke
def test_sampling_token_proposal_config_base_initialization():
    # create base instance to test initialization through TokenProposalConfig
    config = SamplingTokenProposalConfig(
        proposal_type="sampling", speculative_tokens=10, accept_lenience=0.5
    )
    config_dict = config.model_dump()

    # Validate the base class initialization
    config_base = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(config_base, SamplingTokenProposalConfig)
    assert config_base.proposal_type == "sampling"
    assert config_base.speculative_tokens == 10
    assert config_base.accept_lenience == 0.5


@pytest.mark.smoke
def test_sampling_token_proposal_config_nested_initialization():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        sampling_proposal: SamplingTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        sampling_proposals_list: list[SamplingTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        sampling_proposals_dict: dict[str, SamplingTokenProposalConfig]

    parent = ParentModel(
        proposal=SamplingTokenProposalConfig(speculative_tokens=1),
        sampling_proposal=SamplingTokenProposalConfig(speculative_tokens=4),
        proposals_list=[
            SamplingTokenProposalConfig(speculative_tokens=3),
        ],
        sampling_proposals_list=[
            SamplingTokenProposalConfig(speculative_tokens=6),
        ],
        proposals_dict={"first": SamplingTokenProposalConfig(speculative_tokens=2)},
        sampling_proposals_dict={
            "second": SamplingTokenProposalConfig(accept_lenience=0.9)
        },
    )

    # Validate parent model properties
    assert isinstance(parent.proposal, SamplingTokenProposalConfig)
    assert parent.proposal.speculative_tokens == 1
    assert isinstance(parent.sampling_proposal, SamplingTokenProposalConfig)
    assert parent.sampling_proposal.speculative_tokens == 4
    assert isinstance(parent.proposals_list[0], SamplingTokenProposalConfig)
    assert parent.proposals_list[0].speculative_tokens == 3
    assert isinstance(parent.sampling_proposals_list[0], SamplingTokenProposalConfig)
    assert parent.sampling_proposals_list[0].speculative_tokens == 6
    assert isinstance(parent.proposals_dict["first"], SamplingTokenProposalConfig)
    assert parent.proposals_dict["first"].speculative_tokens == 2
    assert isinstance(
        parent.sampling_proposals_dict["second"], SamplingTokenProposalConfig
    )
    assert parent.sampling_proposals_dict["second"].accept_lenience == 0.9


@pytest.mark.smoke
def test_sampling_token_proposal_config_invalid_initialization():
    # Test with invalid proposal_type
    with pytest.raises(ValidationError) as exc_info:
        SamplingTokenProposalConfig(proposal_type="invalid_type")  # type: ignore[arg-type]
    assert "proposal_type" in str(exc_info.value)

    # Test with invalid speculative_tokens (negative value)
    with pytest.raises(ValidationError) as exc_info:
        SamplingTokenProposalConfig(speculative_tokens=-1)
    assert "speculative_tokens" in str(exc_info.value)

    # Test with invalid accept_lenience (negative value)
    with pytest.raises(ValidationError) as exc_info:
        SamplingTokenProposalConfig(accept_lenience=-0.1)
    assert "accept_lenience" in str(exc_info.value)

    # Test with incorrect types
    with pytest.raises(ValidationError) as exc_info:
        SamplingTokenProposalConfig(speculative_tokens="not_an_int")  # type: ignore[arg-type]
    assert "speculative_tokens" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SamplingTokenProposalConfig(accept_lenience="not_a_float")  # type: ignore[arg-type]
    assert "accept_lenience" in str(exc_info.value)


@pytest.mark.smoke
def test_sampling_token_proposal_config_marshalling():
    # Create original config with custom values
    original_config = SamplingTokenProposalConfig(
        speculative_tokens=10, accept_lenience=0.8
    )

    # Convert to dict
    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["proposal_type"] == "sampling"
    assert config_dict["speculative_tokens"] == 10
    assert config_dict["accept_lenience"] == 0.8

    # Recreate from dict using model_validate on base class
    recreated_config = TokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, SamplingTokenProposalConfig)
    assert recreated_config.proposal_type == "sampling"
    assert recreated_config.speculative_tokens == 10
    assert recreated_config.accept_lenience == 0.8

    # Recreate from dict using model_validate on derived class
    recreated_config = SamplingTokenProposalConfig.model_validate(config_dict)
    assert isinstance(recreated_config, SamplingTokenProposalConfig)
    assert recreated_config.proposal_type == "sampling"
    assert recreated_config.speculative_tokens == 10
    assert recreated_config.accept_lenience == 0.8


@pytest.mark.smoke
def test_sampling_token_proposal_config_parent_marshalling():
    class ParentModel(BaseModel):
        proposal: TokenProposalConfig
        sampling_proposal: SamplingTokenProposalConfig
        proposals_list: list[TokenProposalConfig]
        sampling_proposals_list: list[SamplingTokenProposalConfig]
        proposals_dict: dict[str, TokenProposalConfig]
        sampling_proposals_dict: dict[str, SamplingTokenProposalConfig]

    # Create original parent model
    original_parent = ParentModel(
        proposal=SamplingTokenProposalConfig(speculative_tokens=1),
        sampling_proposal=SamplingTokenProposalConfig(speculative_tokens=4),
        proposals_list=[
            SamplingTokenProposalConfig(speculative_tokens=3),
        ],
        sampling_proposals_list=[
            SamplingTokenProposalConfig(speculative_tokens=6),
        ],
        proposals_dict={"first": SamplingTokenProposalConfig(speculative_tokens=2)},
        sampling_proposals_dict={
            "second": SamplingTokenProposalConfig(accept_lenience=0.9)
        },
    )

    # Convert to dict
    parent_dict = original_parent.model_dump()
    parent = ParentModel.model_validate(parent_dict)

    # Validate parent model and dict properties are correct types and match original
    assert isinstance(parent_dict, dict)
    assert isinstance(parent, ParentModel)
    assert isinstance(parent_dict["proposal"], dict)
    assert isinstance(parent.proposal, SamplingTokenProposalConfig)
    assert (
        1
        == parent_dict["proposal"]["speculative_tokens"]
        == parent.proposal.speculative_tokens
    )
    assert isinstance(parent_dict["sampling_proposal"], dict)
    assert isinstance(parent.sampling_proposal, SamplingTokenProposalConfig)
    assert (
        4
        == parent_dict["sampling_proposal"]["speculative_tokens"]
        == parent.sampling_proposal.speculative_tokens
    )
    assert isinstance(parent_dict["proposals_list"], list)
    assert isinstance(parent.proposals_list[0], SamplingTokenProposalConfig)
    assert (
        3
        == parent_dict["proposals_list"][0]["speculative_tokens"]
        == parent.proposals_list[0].speculative_tokens
    )
    assert isinstance(parent_dict["sampling_proposals_list"], list)
    assert isinstance(parent.sampling_proposals_list[0], SamplingTokenProposalConfig)
    assert (
        6
        == parent_dict["sampling_proposals_list"][0]["speculative_tokens"]
        == parent.sampling_proposals_list[0].speculative_tokens
    )
    assert isinstance(parent_dict["proposals_dict"], dict)
    assert isinstance(parent.proposals_dict["first"], SamplingTokenProposalConfig)
    assert (
        2
        == parent_dict["proposals_dict"]["first"]["speculative_tokens"]
        == parent.proposals_dict["first"].speculative_tokens
    )
    assert isinstance(parent_dict["sampling_proposals_dict"], dict)
    assert isinstance(
        parent.sampling_proposals_dict["second"], SamplingTokenProposalConfig
    )
    assert (
        0.9
        == parent_dict["sampling_proposals_dict"]["second"]["accept_lenience"]
        == parent.sampling_proposals_dict["second"].accept_lenience
    )


@pytest.mark.smoke
def test_sampling_token_proposal_config_compiled_loading():
    proposal_dict = {
        "proposal_type": "sampling",
        "speculative_tokens": 10,
        "accept_lenience": 0.8,
    }
    proposal = TokenProposalConfig.model_validate(proposal_dict)
    assert isinstance(proposal, SamplingTokenProposalConfig)
    assert proposal.proposal_type == "sampling"
    assert proposal.speculative_tokens == 10
    assert proposal.accept_lenience == 0.8
