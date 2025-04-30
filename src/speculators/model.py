from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch.nn import Module
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.generation.utils import BaseStreamer, GenerateOutput, GenerationMixin
from transformers.utils import PushToHubMixin

from speculators.config import SpeculatorConfig


class TokenProposal(ABC):
    @abstractmethod
    def init_generation(
        self,
        draft_model: Module,
        verifier_model: Module,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """
        TODO: docs
        """
        ...

    @abstractmethod
    def generate_and_verify_next(
        self,
        draft_model: Module,
        verifier_model: Module,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """
        TODO: docs
        """
        ...


class SpeculatorModel(ABC, Module, GenerationMixin, PushToHubMixin):
    """
    TODO
    """

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        *args,
        **kwargs,
    ) -> "SpeculatorModel":
        """
        TODO: docs
        """
        raise NotImplementedError("from_pretrained is not yet implemented.")

    def __init__(
        self,
        draft_model: Module,
        verifier_model: Module,
        token_proposals: dict[str, TokenProposal],
    ):
        self.draft_model = draft_model
        self.verifier_model = verifier_model
        self.token_proposals = token_proposals

    @abstractmethod
    @property
    def config(self) -> SpeculatorConfig:
        """
        TODO: docs
        """
        ...

    def forward(self, *args, **kwargs):
        """
        TODO: docs -- training
        """
        return self.draft_model(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], list[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional[PreTrainedModel] = None,
        streamer: Optional[BaseStreamer] = None,
        token_proposal_method: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        TODO: docs
        """
        if token_proposal_method is None:
            token_proposal_method = self.config.default_proposal_method

        if token_proposal_method not in self.token_proposals:
            raise ValueError(
                f"Token proposal method {token_proposal_method} not found in "
                f"speculator model. Available methods: {list(self.token_proposals.keys())}"
            )

        # TODO: setup logic

        self.token_proposals[token_proposal_method].init_generation(
            self.draft_model,
            self.verifier_model,
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )

        while True:
            tokens = self.token_proposals[
                token_proposal_method
            ].generate_and_verify_next(
                self.draft_model,
                self.verifier_model,
                inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                assistant_model=assistant_model,
                streamer=streamer,
                **kwargs,
            )
            # streaming if set
            # stopping criteria check

        # return outputs

    def assisted_decoding(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "SpeculatorModel only supports speculative / assisted decoding through "
            "the `generate` method."
        )

    def save(
        self,
        directory: Union[str, Path],
        push_to_hub: bool = False,
        **kwargs,  # noqa: ARG002
    ):
        raise NotImplementedError("SpeculatorModel save is not yet implemented.")
