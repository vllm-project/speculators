"""
A module containing the implementation of the static tree based token proposal method,
where a tree of speculative tokens is generated with a set branching scheme and
depth. Each chain within the tree is verified in parallel by the verifier, increasing
the compute required for verification and speculation, but increasing the liklihood
of matching the verifier's top token. This technique is described in detail in the
[Eagle paper](https://arxiv.org/abs/2401.15077).

Classes:
    - StaticTreeTokenProposalConfig: Configuration for the static tree based token
      proposal method
"""

from typing import Literal

from pydantic import Field

from speculators.config import TokenProposalConfig

__all__ = ["StaticTreeTokenProposalConfig"]


@TokenProposalConfig.register("static_tree")
class StaticTreeTokenProposalConfig(TokenProposalConfig):
    """
    Configuration for the static tree based token proposal method, where a tree of
    speculative tokens is generated with a set branching scheme and depth. Each chain
    within the tree is verified in parallel by the verifier, increasing the compute
    required for verification and speculation, but increasing the liklihood of matching
    the verifier's top token. This technique is described in detail in the
    [Eagle paper](https://arxiv.org/abs/2401.15077).

    The default values for construction of the tree are set to mimic the values
    used in the Eagle paper.
    """

    proposal_type: Literal["static_tree"] = Field(
        default="static_tree",
        description="The type of this token proposal.",
    )
    initial_branching_factor: int = Field(
        default=4,
        description=(
            "The number of tokens to choose from the first step of the draft "
            "which creates the initial number of branches in the tree, "
            "or the initial width of the top of the tree. "
        ),
        ge=1,
    )
    branching_factor: int = Field(
        default=2,
        description=(
            "The number of tokens to choose from each step after the first step "
            "of the draft which creates the number of branches at each node."
        ),
        ge=1,
    )
    depth: int = Field(
        default=5,
        description=(
            "The number of steps to draft tokens for. "
            "This is the depth of the tree, or the number of levels in the tree."
        ),
        ge=1,
    )
