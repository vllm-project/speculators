"""
Base model classes for speculative decoding implementations.

This module provides the foundation for creating speculator models that generate
candidate tokens verified by base models for accelerated inference. The models
extend Hugging Face's PreTrainedModel and GenerationMixin to maintain full
compatibility with the transformers ecosystem while adding speculative decoding
capabilities, automatic model registration and discovery, dynamic model loading
based on configuration, and flexible verifier attachment.
"""

from __future__ import annotations

import os
from typing import Any, Callable, ClassVar, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput

from speculators.config import SpeculatorModelConfig, VerifierConfig
from speculators.utils import RegistryMixin

__all__ = ["SpeculatorModel", "reload_and_populate_models"]


class SpeculatorModel(  # type: ignore[misc]
    RegistryMixin[type["SpeculatorModel"]], PreTrainedModel, GenerationMixin
):
    """
    Abstract base class for all speculator models.

    Provides the foundation for implementing speculative decoding models that
    generate candidate tokens verified by base verifier models. Combines Hugging Face's
    PreTrainedModel and GenerationMixin with automatic model registration and discovery
    capabilities. All concrete implementations must inherit from this class,
    register with `SpeculatorModel.register(NAME)`, and implement the abstract
    forward method.

    Example:
        ::
            # Load a speculator model with automatic class resolution
            model = SpeculatorModel.from_pretrained("path/to/speculator")

            # Optionally attach a new verifier model
            verifier = AutoModel.from_pretrained("path/to/verifier")
            model.attach_verifier(verifier)

            # Generate with speculative decoding
            outputs = model.generate(input_ids, max_length=100)

    :cvar auto_package: Package path for automatic model discovery
    :cvar registry_auto_discovery: Whether to enable automatic registry population
    :cvar config_class: Configuration class for speculator models
    :cvar base_model_prefix: Prefix for model parameter names
    :cvar main_input_name: Primary input tensor name
    :cvar _keys_to_ignore_on_load_missing: Model parameter keys to ignore when missing
        during loading
    """

    # Registry configuration
    auto_package: ClassVar[str] = "speculators.models"
    registry_auto_discovery: ClassVar[bool] = True

    # PreTrainedModel settings
    config_class: ClassVar[type[SpeculatorModelConfig]] = SpeculatorModelConfig  # type: ignore[assignment,misc]
    base_model_prefix: ClassVar[str] = "model"  # type: ignore[misc]
    main_input_name: ClassVar[str] = "input_ids"  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[assignment,misc]
        "verifier*",
    ]

    @classmethod
    def from_pretrained(
        cls: type[SpeculatorModel],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        verifier_attachment_mode: Literal["detached", "full", "train_only"] | None = (
            None
        ),
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpeculatorModel:
        """
        Load a pretrained speculator model from Hub or local directory.

        Automatically resolves the correct speculator model class based on configuration
        type and loads the model with appropriate weights. If called on the base
        SpeculatorModel class, automatically determines and instantiates the correct
        subclass based on the model configuration.

        :param pretrained_model_name_or_path: Model identifier on Hugging Face Hub
            or path to local directory containing model files
        :param model_args: Additional positional arguments passed to model constructor
        :param verifier: Optional verifier model to attach for speculative decoding
        :param verifier_attachment_mode: How verifier is attached ("detached", "full",
            "train_only")
        :param config: Model configuration instance, path to config file, or None
        :param cache_dir: Directory to cache downloaded files
        :param ignore_mismatched_sizes: Whether to ignore size mismatches when loading
        :param force_download: Whether to force re-download of model files
        :param local_files_only: Whether to avoid downloading and only use local cache
        :param token: Authentication token for private models on Hugging Face Hub
        :param revision: Model revision to load (branch name, tag, or commit hash)
        :param use_safetensors: Whether to use safetensors format for loading weights
        :param weights_only: Whether to only load model weights without optimizer
            states
        :param kwargs: Additional keyword arguments passed to model constructor
        :return: SpeculatorModel instance of appropriate subclass with pretrained
            weights
        """
        if not config:
            if not pretrained_model_name_or_path:
                raise ValueError(
                    "Either `config` or `pretrained_model_name_or_path` must be "
                    "provided to load a SpeculatorModel."
                )
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )

        if not isinstance(config, SpeculatorModelConfig):
            # once conversion is added, need to handle the case where a non speculator
            # config is passed in as a kwarg and auto convert
            raise TypeError(
                f"Expected config to be an instance of SpeculatorModelConfig, "
                f"got {type(config)}."
            )

        if not pretrained_model_name_or_path and not kwargs.get("state_dict"):
            raise ValueError(
                "Either `pretrained_model_name_or_path` or `state_dict` must be "
                "provided to load a SpeculatorModel."
            )

        if cls is SpeculatorModel:
            # generic call to from_pretrained on this class, need to resolve the
            # specific model class to use for loading based on the config and registry
            model_class = cls.registered_model_class_from_config(config)
            return model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                verifier=verifier,
                verifier_attachment_mode=verifier_attachment_mode,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                weights_only=weights_only,
                **kwargs,
            )

        return super().from_pretrained(  # type: ignore[misc]
            pretrained_model_name_or_path,
            *model_args,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )

    @classmethod
    def registered_model_class_from_config(
        cls, config: SpeculatorModelConfig
    ) -> type[SpeculatorModel]:
        """
        Look up the appropriate speculator model class from registry based on config
        type.

        Matches the config class to the corresponding model class that was registered
        during auto-discovery or manual registration.

        :param config: Configuration instance to find registered model class for
        :return: Registered model class that matches the configuration type
        """
        if not isinstance(config, SpeculatorModelConfig):
            raise TypeError(
                f"Expected config to be an instance of SpeculatorModelConfig, "
                f"got {type(config)} {config}."
            )

        if type(config) is SpeculatorModelConfig:
            raise TypeError(
                "Received a SpeculatorModelConfig instance but expected a subclass. "
                "Use the specific subclass of SpeculatorModelConfig instead. "
                f"Received: {type(config)} {config}"
            )

        if not cls.registry:
            raise ValueError(
                "No registered model classes found. "
                "Ensure that models are registered with "
                "`SpeculatorModel.register(NAME)` or that auto-discovery is enabled."
            )

        for _, model_class in cls.registry.items():
            model_config_class: type[SpeculatorModelConfig] = model_class.config_class

            if type(config) is model_config_class:
                return model_class

        raise ValueError(
            f"No registered model class found for config type {type(config)}. "
            f"Available registered model classes: {list(cls.registry.keys())}."
        )

    def __init__(
        self,
        config: SpeculatorModelConfig,
        verifier: str | os.PathLike | PreTrainedModel | None,
        verifier_attachment_mode: Literal["detached", "full", "train_only"] | None,
        **kwargs,
    ):
        """
        Initialize a SpeculatorModel instance.

        Sets up the basic structure for a speculator model, including configuration
        storage and optional verifier model attachment for speculative decoding
        validation. If no verifier is provided during initialization, it must be
        attached later using the attach_verifier method before calling generate.

        :param config: Configuration instance containing model hyperparameters and
            speculative decoding settings
        :param verifier: Verifier model to attach for speculative decoding validation
        :param verifier_attachment_mode: How verifier is attached ("detached", "full",
            "train_only")
        :param kwargs: Additional keyword arguments passed to PreTrainedModel
            constructor
        """
        if not config:
            raise ValueError(
                "Config must be provided to initialize a SpeculatorModel. "
                "Use SpeculatorModelConfig to create a valid configuration."
            )

        if not isinstance(config, SpeculatorModelConfig):
            raise TypeError(
                f"Expected config to be an instance of SpeculatorModelConfig, "
                f"got {type(config)} {config}."
            )

        super().__init__(config, **kwargs)
        self.config: SpeculatorModelConfig = config
        self.verifier: PreTrainedModel | None = None
        self.verifier_attachment_mode: Literal["detached", "full", "train_only"] = (
            "detached"
        )

        verifier = verifier or config.speculators_config.verifier.name_or_path
        if verifier is not None and verifier_attachment_mode != "detached":
            self.attach_verifier(verifier, mode=verifier_attachment_mode)

    def resolve_verifier(
        self, verifier: str | os.PathLike | PreTrainedModel
    ) -> PreTrainedModel:
        """
        Resolve the verifier model from a given path or identifier.

        Loads the verifier model from a specified path or identifier, ensuring
        compatibility with the speculator's configuration.

        :param verifier: Verifier model path, identifier, or PreTrainedModel instance
        :return: Resolved PreTrainedModel instance for the verifier
        """
        if not verifier:
            raise ValueError(
                "Verifier must be provided as a path, identifier, or PreTrainedModel. "
            )

        if not isinstance(verifier, (str, os.PathLike, PreTrainedModel)):
            raise TypeError(
                f"Expected verifier to be a PreTrainedModel, a string path, "
                f"or an os.PathLike object, got {type(verifier)} {verifier}."
            )

        if isinstance(verifier, PreTrainedModel):
            return verifier

        return AutoModelForCausalLM.from_pretrained(verifier)

    def attach_verifier(
        self,
        verifier: str | os.PathLike | PreTrainedModel,
        mode: Literal["full", "train_only"] | None = None,
        add_to_config: bool = True,
    ) -> PreTrainedModel:
        """
        Attach a verifier model for speculative decoding validation.

        Attaches a verifier model that validates candidate tokens generated by the
        speculator during speculative decoding. The verifier should be compatible
        with the speculator's configuration in terms of vocabulary, architecture,
        and tokenization.

        :param verifier: Verifier model path, identifier, or PreTrainedModel instance
        :param mode: Attachment mode ("full" or "train_only")
        :param add_to_config: Whether to add verifier references to speculator config
        :return: PreTrainedModel instance for the attached verifier
        """
        if self.verifier_attachment_mode != "detached":
            raise RuntimeError(
                "Cannot attach a verifier when the speculator is not in detached mode. "
                "Detach the current verifier first using `detach_verifier()`."
            )

        if mode not in {"full", "train_only", None}:
            raise ValueError(
                f"Invalid verifier_attachment_mode: {mode}. "
                "Must be one of 'full', 'train_only', or None."
            )

        verifier = self.resolve_verifier(verifier)
        self.verifier_attachment_mode = mode or "full"
        self.verifier = (
            verifier if self.verifier_attachment_mode == "full" else None
        )  # Expect subclasses to handle references if train_only

        if add_to_config:
            try:
                self.config.speculators_config.verifier = VerifierConfig.from_pretrained(
                    verifier
                )
            except (OSError, ValueError, Exception) as e:
                raise RuntimeError(
                    f"Failed to load verifier configuration from '{verifier}': {e}"
                ) from e

        return verifier

    def detach_verifier(self):
        """
        Remove reference to attached verifier model and free associated memory.

        After calling this method, the speculator will not be able to perform
        forward passes or generation until a new verifier is attached.
        """
        if self.verifier_attachment_mode == "detached":
            raise RuntimeError(
                "Verifier is already detached, cannot be called again until "
                "a new verifier is attached."
            )

        if self.verifier is not None:
            del self.verifier

        self.verifier = None
        self.verifier_attachment_mode = "detached"

    def state_dict(
        self,
        *,
        destination: dict[str, Any] = None,  # type: ignore[assignment]
        prefix: str = "",
        keep_vars: bool = False,
    ):
        """
        Get state dictionary excluding verifier model parameters.

        Overrides PyTorch's state_dict method to ensure save pathways within
        Transformers PreTrainedModel do not include the verifier model's parameters,
        allowing the speculator model to be saved and loaded without including
        the verifier's state.

        :param destination: Optional dictionary to store the state
        :param prefix: Optional prefix for parameter names
        :param keep_vars: Whether to keep Variables in the state_dict
        :return: Dictionary containing speculator model state excluding verifier
            parameters
        """
        tmp_verifier = self.verifier
        self.verifier = None
        state = super().state_dict(  # type: ignore[misc]
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        self.verifier = tmp_verifier

        return state

    def forward(self, *args, **kwargs):
        """
        Define forward pass computation for the speculator model.

        Must be implemented by all concrete speculator model subclasses. Defines
        how the model processes inputs to generate candidate tokens or logits
        specifically for training pipelines. Use `model.generate` for generation
        tasks, which will handle speculative decoding with the attached verifier.

        :param args: Positional arguments for forward pass (input_ids, attention_mask,
            position_ids, etc.)
        :param kwargs: Keyword arguments for forward pass with model-specific parameters
        :return: Model outputs including logits or candidate token sequences
        """
        raise NotImplementedError(
            "The forward method is only supported on concrete "
            "speculator model subclasses."
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,  # noqa: ARG002
        generation_config: GenerationConfig | None = None,  # noqa: ARG002
        logits_processor: LogitsProcessorList | None = None,  # noqa: ARG002
        stopping_criteria: StoppingCriteriaList | None = None,  # noqa: ARG002
        prefix_allowed_tokens_fn: (  # noqa: ARG002
            Callable[[int, torch.Tensor], list[int]] | None
        ) = None,
        synced_gpus: bool | None = None,  # noqa: ARG002
        assistant_model: PreTrainedModel | None = None,  # type: ignore[override]  # noqa: ARG002
        streamer: BaseStreamer | None = None,  # noqa: ARG002
        negative_prompt_ids: torch.Tensor | None = None,  # noqa: ARG002
        negative_prompt_attention_mask: torch.Tensor | None = None,  # noqa: ARG002
        use_model_defaults: bool | None = None,  # noqa: ARG002
        custom_generate: str | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> GenerateOutput | torch.LongTensor:
        """
        Generate text using speculative decoding with attached verifier model.

        Follows the standard transformers generation interface, making it compatible
        with existing generation workflows while adding speculative decoding
        capabilities for faster generation.

        :param inputs: Input token IDs to generate from
        :param generation_config: Configuration for generation parameters
        :param logits_processor: List of logits processors to apply during generation
        :param stopping_criteria: List of stopping criteria to determine when to stop
        :param prefix_allowed_tokens_fn: Function to constrain generation to allowed
            tokens based on current prefix
        :param synced_gpus: Whether to synchronize GPUs during distributed generation
        :param assistant_model: Assistant model for generation compatibility
        :param streamer: Streamer to output tokens as they are generated
        :param negative_prompt_ids: Token IDs for negative prompting
        :param negative_prompt_attention_mask: Attention mask for negative prompt tokens
        :param use_model_defaults: Whether to use model-specific default parameters
        :param custom_generate: Custom generation parameter
        :param kwargs: Additional keyword arguments for generation
        :return: Generated token sequences as GenerateOutput object or LongTensor
        """
        if self.verifier is None:
            raise ValueError(
                "Verifier model is not attached. Please attach a verifier model "
                "before calling generate."
            )

        raise NotImplementedError(
            "The generate method for speculator models is not implemented yet."
        )


def reload_and_populate_models():
    """
    Trigger automatic discovery and registration of all SpeculatorModel subclasses.

    Discovers and registers all SpeculatorModel subclasses found in the
    speculators.models package that have been registered with
    `SpeculatorModel.register(NAME)`. Enables dynamic model loading and
    instantiation based on configuration types without requiring explicit imports.
    """
    SpeculatorModel.auto_populate_registry()
