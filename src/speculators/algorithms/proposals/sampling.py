from typing import Optional

import torch

from speculators.base.config import TokenProposalConfig
from speculators.base.objects import SpeculatorModel, TokenProposal


class SamplingTokenProposal(TokenProposal):
    """
    Implementation for the sampling token proposal algorithm where a sampling
    temperature is used to select the next token for each proposal from the drafter.
    After X number of tokens, the verifier is called to to validate the proposed tokens.
    """

    @classmethod
    def from_config(cls, config: TokenProposalConfig) -> "SamplingTokenProposal":
        """
        Create a sampling token proposal instance from the provided config.

        :param config: The configuration for the sampling token proposal.
        :return: The instance of the sampling token proposal.
        """
        # Placeholder, need to define and pull args from config
        return cls(config)  # type: ignore[arg-type,call-arg]

    def __init__(self, **kwargs):
        """
        Initialize the sampling token proposal with the provided arguments.

        :param kwargs: Additional arguments for the sampling token proposal.
            Need to define exact arguments for the implementation.
        """
        raise NotImplementedError(
            "SamplingTokenProposal initialization is not implemented yet."
        )

    @property
    def config(self) -> TokenProposalConfig:
        """
        Get the configuration of the sampling token proposal.

        :return: The configuration of the sampling token proposal.
        """
        return TokenProposalConfig(
            type_="sampling",
            args={...},  # type: ignore[arg-type]
        )

    def init_generation(
        self,
        input_ids: torch.Tensor,
        speculator: SpeculatorModel,
        **kwargs,
    ) -> tuple[torch.LongTensor, Optional[torch.FloatTensor], dict]:
        """
        The first stage for the token proposal algorithm that handles initial setup of
        state, the prefill run through the verifier model to generate the first token,
        and setup for state management for following tokens such as KV cache management
        and intermediate outputs from the verifier model.

        :param input_ids: The input IDs representing the initial prompt to generate
            the initial token for and setup state based on.
        :param speculator: The speculator model instance that includes the verifier
            and draft models.
        :param kwargs: Additional keyword arguments for the generation call.
            This is still in process and more named arguments will be added to the
            function signature rather than keeping them in kwargs.
        :return: A tuple containing the input and generated token IDs,
            optional logis for the generated token,
            and a dict containing any state that should be preserved and passed to the
            next generate_and_verify_next call.
        """
        raise NotImplementedError(
            "SamplingTokenProposal init_generation is not implemented yet."
        )

    def generate_and_verify_next(
        self,
        input_ids: torch.Tensor,
        logits: Optional[torch.FloatTensor],
        state: dict,
        speculator: SpeculatorModel,
        **kwargs,
    ) -> Optional[tuple[torch.LongTensor, Optional[torch.FloatTensor], dict]]:
        """
        The second stage for the token proposal algorithm that handles generating
        the next set of candidate tokens, the verification of the candidate tokens,
        and management of any state that is needed such as the KV cache and intermediate
        outputs from the verifier model.

        :param input_ids: The input IDs representing all tokens generated so far.
        :param logits: The optional logits for the generated tokens so far.
        :param state: The state dict containing any state that should be preserved
            and passed to the next generate_and_verify_next call.
        :param speculator: The speculator model instance that includes the verifier
            and draft models.
        :param kwargs: Additional keyword arguments for the generation call.
            This is still in process and more named arguments will be added to the
            function signature rather than keeping them in kwargs.
        :return: A tuple containing the input and generated token IDs,
            optional logits for the generated token,
            and a dict containing any state that should be preserved and passed to the
            next generate_and_verify_next call. Once generation is complete and the
            stopping criteria is met, None is returned.
        """
        raise NotImplementedError(
            "SamplingTokenProposal generate_and_verify_next is not implemented yet."
        )
