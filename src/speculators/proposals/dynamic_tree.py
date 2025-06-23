"""
A module containing the implementations of the dynamic tree based token proposal method,
where the top tokens from the speculator based on the speculator's probabilities and the
expansion_factor are used to create the next tokens, limiting the growth of the tree and
focusing on the most likely tokens. After drafting to the selected depth, the tree is
pruned down to the top speculative_tokens tokens based on the product of their
probabilities up to that point in the tree for verification.
This is described in more detail in the [Eagle 2 paper](https://arxiv.org/abs/2406.16858).

Classes:
    - DynamicTreeTokenProposalConfig: Configuration for the dynamic tree based token
      proposal method
"""

from typing import Literal

from pydantic import Field

from speculators.config import TokenProposalConfig

__all__ = ["DynamicTreeTokenProposalConfig"]


@TokenProposalConfig.register("dynamic_tree")
class DynamicTreeTokenProposalConfig(TokenProposalConfig):
    """
    Configuration for the dynamic tree based token proposal method, where the top tokens
    from the speculator based on the speculator's probabilities and the expansion_factor
    are used to create the next tokens, limiting the growth of the tree and focusing on
    the most likely tokens. After drafting to the selected depth, the tree is pruned
    down to the top speculative_tokens tokens based on the product of their
    probabilities up to that point in the tree for verification.
    This is described in more detail in the [Eagle 2 paper](https://arxiv.org/abs/2406.16858).

    The default values for construction of the tree are set to mimic the values
    used in the Eagle 2 paper.
    """

    proposal_type: Literal["dynamic_tree"] = Field(
        default="dynamic_tree",
        description="The type of this token proposal.",
    )
    speculative_tokens: int = Field(
        default=48,
        description=(
            "The number of tokens created by the speculator to run through the "
            "verifier on each forward pass. This is the number of tokens the tree is "
            "pruned down to from the full tree created by the speculator. "
            "Tokens are ranked based on the product of their probabilities up to "
            "that point in the tree. If a token is discarded, all tokens after it "
            "in the tree are also discarded."
        ),
        ge=1,
    )
    expansion_factor: int = Field(
        default=10,
        description=(
            "The number of tokens, or branches, to create at each step for the "
            "speculator. The most likely tokens equalling the expansion_factor "
            "number and based on the speculator's probabilities, are chosen to run "
            "through the speculator at each step. For the initial step, this is the "
            "number of tokens selected from the speculator's top tokens."
        ),
        ge=1,
    )
    depth: int = Field(
        default=6,
        description=(
            "The number of steps to draft tokens for. "
            "This is the depth of the tree, or the number of levels in the tree."
        ),
        ge=1,
    )
