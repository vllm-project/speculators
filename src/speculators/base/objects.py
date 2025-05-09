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
from transformers.generation.utils import (
    BaseStreamer,
    CandidateGenerator,
    GenerateOutput,
    GenerationMixin,
)
from transformers.utils import PushToHubMixin

from speculators.base.config import (
    DraftModelConfig,
    SpeculatorConfig,
    TokenProposalConfig,
)
from speculators.utils import convert_to_speculators, detect_model_format

__all__ = ["Drafter", "SpeculatorModel", "TokenProposal"]


class Drafter(ABC, Module):
    @classmethod
    def from_config(cls, config: DraftModelConfig) -> "Drafter":
        """
        Create the supported specaultors draft model from the provided config.

        :param config: The configuration for the FFN drafter.
        :return: The module instance of the FFN drafter.
        """
        # need to implement factory method to create the drafter
        raise NotImplementedError(
            "Drafter from_config is not implemented. "
            "Please use the specific drafter class."
        )

    @abstractmethod
    @property
    def config(self) -> DraftModelConfig:
        """
        The config object for the drafter model.
        This must be a subclass of DraftModelConfig and overwritten
        in the implementing class.

        :return: The config object for the drafter model.
        """
        ...


class TokenProposal(ABC):
    """
    Base class for how the speculator generates and validates the draft tokens.
    Inherited classes are expected to implement
        - init_generation which handles initial setup of state, the prefill run through
          the verifier model to generate the first token, and setup for state management
          for following tokens such as KV cache management.
        - generate_and_verify_next which handles generating the next set of candidate
          tokens, the verification of the candidate tokens, and management of any state
          that is needed such as the KV cache.
    """

    @classmethod
    def from_config(cls, config: TokenProposalConfig) -> "TokenProposal":
        """
        Create a token proposal instance from the provided config.

        :param config: The configuration for the token proposal.
        :return: The instance of the token proposal.
        """
        # need to implement factory method to create the token proposal
        raise NotImplementedError(
            "TokenProposal from_config is not implemented. "
            "Please use the specific token proposal class."
        )

    @abstractmethod
    @property
    def config(self) -> TokenProposalConfig:
        """
        The config object for the token proposal.
        This must be a subclass of TokenProposalConfig and overwritten
        in the implementing class.

        :return: The config object for the token proposal.
        """
        ...

    @abstractmethod
    def init_generation(
        self,
        input_ids: torch.Tensor,
        speculator: "SpeculatorModel",
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
        ...

    @abstractmethod
    def generate_and_verify_next(
        self,
        input_ids: torch.Tensor,
        logits: Optional[torch.FloatTensor],
        state: dict,
        speculator: "SpeculatorModel",
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
        ...


class SpeculatorModel(ABC, Module, GenerationMixin, PushToHubMixin):
    """
    Base class for all speculator algorithm implementations.
    It handles defining a standardized interface for loading, saving, and using
    speculator models for both training and inference.
    """

    @staticmethod
    def from_pretrained(
        source: Union[str, Path, Module],
        config: Optional[Union[str, Path, dict, SpeculatorConfig]] = None,
        verifier: Optional[Union[str, Path, Module]] = "unset",
        **kwargs,
    ) -> "SpeculatorModel":
        """
        Load a speculator model from a pretrained model from various sources into a
        speculator.algorithms implementation. Specifically, it loads or converts the
        config, instantiates the correct speculator model class, and loads/sets the
        parameters for the nested properties within the speculator instance.
        This can include:
        - A speculator model stored on the Hugging Face Hub.
        - A speculator model stored locally in a directory.
        - A speculative decoding model from another library with conversion support
          for either the file formats or a model instance:
            - Eagle repo model
            - Haas repo model
            - Eagle3 repo model

        :param source: A Hugging Face Hub model ID, a local directory containing a
            speculators config and model files, a local directory or file containing
            a model from another supported library, or a model instance from another
            supported library.
        :param config: The path to the config file or a config object.
            If not provided, the config will be loaded from the source.
        :param verifier: The HuggingFace model ID, local directory,
            or a PyTorch module representing the verifier model.
            If not provided, the verifier model will be loaded from the source,
            if available. If set to None, the verifier model will not be loaded.
        :param kwargs: Additional keyword arguments for loading the model.
            This is still in process and more named arguments will be added to the
            function signature rather than keeping them in kwargs as needed.
        """
        # Branch pathways to be implemented:
        if is_huggingface_id := False:  # fill in with logic to detect Hugging Face ID
            raise NotImplementedError(
                "Loading from Hugging Face ID is not yet implemented."
            )

        if (format_ := detect_model_format(source, config)) != "speculators":
            return convert_to_speculators(
                source=source,
                config=config,
                verifier=verifier,
                format_=format_,
                **kwargs,
            )

        speculator = (
            ...
        )  # factory method to create the appropriate speculator algorithm
        # load weights from the source into the speculator instance

        return speculator

    def __init__(
        self,
        drafter: Module,
        verifier: Optional[Module],
        proposals: dict[str, TokenProposal],
    ):
        """
        Constructor for the SpeculatorModel class.
        All speculator implementations must inherit from this class, setup their
        state as needed, and call this constructor with the draft model, verifier model,
        and token proposals to enable general functionality for the speculator.

        :param drafter: The draft model to use for the speculator.
        :param verifier: The verifier model to use for the speculator.
        :param proposals: A dictionary of the supported token proposal methods
            for the speculator and their configurations, hyperparameters, and defaults.
            The keys are the method names, and the values are the configurations.
            This is used to resolve the token proposal method to use for the speculator
            implementation.
        """
        self.drafter = drafter
        self.verifier = verifier  # need to make optional and attach later
        self.proposals = proposals

    @abstractmethod
    @property
    def config(self) -> SpeculatorConfig:
        """
        The config object for the speculator model.
        This must be a subclass of SpeculatorConfig and overwritten
        in the implementing class.

        :return: The config object for the speculator model.
        """
        ...

    def attach_verifier(self, verifier: Module):
        """
        Attach the verifier model to the speculator model.
        This is used to set the verifier model after the speculator model has been
        instantiated.

        :param verifier: The verifier model to attach to the speculator model.
        """
        self.detach_verifier()
        self.verifier = verifier

    def detach_verifier(self):
        """
        Detach the verifier model from the speculator model.
        This is used to remove the verifier model and any state associated with it
        from the speculator model.
        """
        self.verifier = None

    def forward(self, *args, **kwargs):
        """
        The forward method used for training the draft model.
        This can optionally be overwritten in the implementing class,
        by default it passes args and kwargs to the draft model.

        :param args: The positional arguments to pass to the draft model.
        :param kwargs: The keyword arguments to pass to the draft model.
        :return: The output of the draft model.
        """
        return self.drafter(*args, **kwargs)

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
        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            token_proposal_method=token_proposal_method,
            **kwargs,
        )

    def save(
        self,
        directory: Union[str, Path],
        push_to_hub: bool = False,
        **kwargs,  # noqa: ARG002
    ):
        raise NotImplementedError("SpeculatorModel save is not yet implemented.")

    def resolve_token_proposal(
        self, method: Optional[Union[str, TokenProposal]]
    ) -> TokenProposal:
        """
        Resolves the token proposal method to use for the speculator implementation.
        Specifically, the instance of the TokenProposal class that is used to generate
        and verify tokens.

        :param method: The token proposal method to use. This can be None, a string
            representing the method name, or an instance of the TokenProposal class.
            If it is a TokenProposal instance, it is returned as is.
            If it is a string, it is used to look up the corresponding
            TokenProposal instance in the proposals dictionary.
            If it is None, the default proposal method from the config is used,
            if available.
            Otherwise, the first method in the proposals dictionary is used.
        :return: The TokenProposal instance to use for generating and verifying tokens.
        :raises ValueError: If the method is not a token proposal method and
            if the method is not found in the proposals dictionary
            or the proposals dictionary is empty.
        """
        if method and isinstance(method, TokenProposal):
            return method

        if not self.proposals:
            raise ValueError(
                "No token proposal methods available. Please provide a method or "
                "ensure the speculator model has token proposal methods defined."
            )

        method = (
            method
            or self.config.default_proposal_method
            or list(self.proposals.keys())[0]
        )

        if method not in self.proposals:
            raise ValueError(
                f"Token proposal method {method} not found in speculator model. "
                f"Available methods: {list(self.proposals.keys())}"
            )

        return self.proposals[method]

    def _prepare_generation_config(
        self,
        generation_config: Optional[GenerationConfig],
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> tuple[GenerationConfig, dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_config.use_cache = True
        generation_config.assistant_early_exit = True  # trigger assisted gen pathway

    def _get_candidate_generator(
        self,
        *args,
        **kwargs,
    ) -> Optional[CandidateGenerator]:
        return None  # not used since we take over the generation process

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: Optional[CandidateGenerator],
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional[BaseStreamer],
        token_proposal_method: Optional[Union[str, TokenProposal]] = None,
        **model_kwargs,
    ):
        token_proposal = self.resolve_token_proposal(token_proposal_method)
        # need to implement any state setup for the general decode
        input_ids, logits, state = token_proposal.init_generation(
            input_ids=input_ids,
            speculator=self,
            # need to determine what else should be passed
        )
        should_stop = False

        while not should_stop:
            input_ids, logits, state = token_proposal.generate_and_verify_next(
                input_ids=input_ids,
                logits=logits,
                state=state,
                speculator=self,
                # need to determine what else should be passed
            )

        return input_ids  # need to return the correct output type and data
