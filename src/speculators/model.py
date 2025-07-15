"""
Base model classes for the Speculators library.

This module contains the base model classes for speculative decoding implementations
in the Speculators library. These classes provide the foundation for creating
speculator models that can perform speculative token generation with verifier
models for accelerated inference.

The models extend Hugging Face's PreTrainedModel and GenerationMixin to maintain
full compatibility with the transformers ecosystem while adding speculative
decoding capabilities. They support automatic model registration and discovery,
dynamic model loading based on configuration, and flexible verifier attachment.

Classes:
    SpeculatorModel: Abstract base class for all speculator models with transformers
        compatibility, automatic registry support, and speculative generation methods

Functions:
    reload_and_populate_models: Automatically populates the model registry for
        discovery and instantiation of registered speculator models
"""

import os
from typing import Any, Callable, ClassVar, Literal, Optional, Union

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
from speculators.utils import ClassRegistryMixin


class SpeculatorModel(ClassRegistryMixin, PreTrainedModel, GenerationMixin):  # type: ignore[misc]
    """
    Abstract base class for all speculator models in the Speculators library.

    This class provides the foundation for implementing speculative decoding models
    that can generate candidate tokens to be verified by a base verifier model.
    It combines the functionality of Hugging Face's PreTrainedModel and GenerationMixin
    with automatic model registration and discovery capabilities.
    All concrete speculator model implementations must inherit from this class,
    register with `SpeculatorModel.register(NAME)`, and
    implement the abstract forward method.

    Example:
        ```python
        # Load a speculator model with automatic class resolution
        model = SpeculatorModel.from_pretrained("path/to/speculator")

        # Optionally attach a new verifier model
        verifier = AutoModel.from_pretrained("path/to/verifier")
        model.attach_verifier(verifier)

        # Generate with speculative decoding
        outputs = model.generate(input_ids, max_length=100)
        ```
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
        cls: type["SpeculatorModel"],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full", "train_only"]
        ] = None,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> "SpeculatorModel":
        """
        Load a pretrained speculator model from the Hugging Face Hub or local directory.

        This method automatically resolves the correct speculator model class based on
        the configuration type and loads the model with the appropriate weights. If
        called on the base SpeculatorModel class, it will automatically determine and
        instantiate the correct subclass based on the model configuration.

        Example:
            ```python
            # Load with automatic class resolution
            model = SpeculatorModel.from_pretrained("RedHatAI/speculator-llama-7b")

            # Load from local directory
            model = SpeculatorModel.from_pretrained("./my_speculator")

            # Load with custom config
            config = SpeculatorModelConfig.from_pretrained("RedHatAI/eagle-llama-7b")
            model = SpeculatorModel.from_pretrained(
                None, config=config, state_dict=state_dict
            )
            ```

        :param pretrained_model_name_or_path: The model identifier on Hugging Face Hub,
            or path to a local directory containing the model files. Can be None if
            config is provided as a path.
        :param model_args: Additional positional arguments passed to the model
            constructor.
        :param verifier: Optional verifier model to attach the speculator to.
            Can be a path to a local model directory, a Hugging Face model identifier,
            or an instance of PreTrainedModel. If provided, the speculator will use this
            verifier for speculative decoding. If None, the speculator will load the
            verifier from the config if specified, or it must be attached later
            using the `attach_verifier` method.
        :param verifier_attachment_mode: Optional mode for how the verifier is
            attached to the speculator. If "detached", any verifier passed in or
            resolved from the config will not be ignored.
            If "full", the verifier is fully integrated into the
            speculator's forward pass and generation methods.
            If "train_only", only the portions of the verifier needed for training
            are attached, allowing for better resource utilization during training.
            If None and a verifier is provided, it defaults to "full".
            If a verifier is not provided and None is found in the config,
            this parameter is ignored.
        :param config: Optional configuration for the model. Can be a
            SpeculatorModelConfig instance, a path to a config file, or None to load
            from model directory.
        :param cache_dir: Directory to cache downloaded files. If None, uses default
            transformers cache directory.
        :param ignore_mismatched_sizes: Whether to ignore size mismatches when loading
            pretrained weights. Useful for loading models with different architectures.
        :param force_download: Whether to force re-download of model files even if
            they exist in cache.
        :param local_files_only: Whether to avoid downloading files and only use local
            cached files. Raises an error if files are not found locally.
        :param token: Optional authentication token for accessing private models on
            Hugging Face Hub. Can be a string token or True to use saved token.
        :param revision: The specific model revision to load (branch name, tag, or
            commit hash). Defaults to "main".
        :param use_safetensors: Whether to use safetensors format for loading weights.
            If None, automatically detects the available format.
        :param weights_only: Whether to only load model weights without optimizer
            states or other training artifacts.
        :param kwargs: Additional keyword arguments passed to the model constructor
            and loading process.
        :return: A SpeculatorModel instance of the appropriate subclass, loaded with
            the pretrained weights and configuration.
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
    ) -> type["SpeculatorModel"]:
        """
        Looks up the appropriate speculator model class from the registry
        based on the configuration type. It matches the config class to the
        corresponding model class that was registered during auto-discovery or manual
        registration.

        :param config: The configuration for which to find the registered model class.
            Must be an instance of a SpeculatorModelConfig subclass.
        :return: The registered model class that matches the configuration type.
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
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]],
        verifier_attachment_mode: Optional[Literal["detached", "full", "train_only"]],
        **kwargs,
    ):
        """
        Initialize a SpeculatorModel instance.

        Sets up the basic structure for a speculator model, including configuration
        storage and optional verifier model attachment. The verifier model is used
        during speculative decoding to validate the tokens proposed by the speculator.

        If no verifier is provided during initialization, it must be attached later
        using the attach_verifier method before calling generate.

        :param config: The configuration for the speculator model. Must be a
            SpeculatorModelConfig instance containing model hyperparameters and
            speculative decoding settings.
        :param verifier: The verifier model to attach. This can be a path to a local
            model directory, a Hugging Face model identifier, or an instance of
            PreTrainedModel. If provided, the speculator will use this verifier for
            speculative decoding. If None, the speculator will load the verifier from
            the config if specified, or it must be attached later using the
            `attach_verifier` method.
        :param verifier_attachment_mode: Optional mode for how the verifier is
            attached to the speculator. If "detach", any verifier passed in or
            resolved from the config will not be attached.
            If "full", the verifier is fully integrated into the
            speculator's forward pass and generation methods.
            If "train_only", only the portions of the verifier needed for training
            are attached, allowing for better resource utilization during training.
            If None and a verifier is provided, it defaults to "full".
            If a verifier is not provided and None is found in the config,
            this parameter is ignored.
        :param kwargs: Additional keyword arguments passed to the parent
            PreTrainedModel constructor.
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
        self.verifier: Optional[PreTrainedModel] = None
        self.verifier_attachment_mode: Literal["detached", "full", "train_only"] = (
            "detached"
        )

        verifier = verifier or config.speculators_config.verifier.name_or_path
        if verifier is not None and verifier_attachment_mode != "detached":
            self.attach_verifier(verifier, mode=verifier_attachment_mode)

    def resolve_verifier(
        self, verifier: Union[str, os.PathLike, PreTrainedModel]
    ) -> PreTrainedModel:
        """
        Resolves the verifier model from a given path or identifier.

        This method loads the verifier model from a specified path or identifier,
        ensuring it is compatible with the speculator's configuration. If the
        verifier is already attached, it returns the existing verifier instance.

        :param verifier: The verifier model to resolve. Can be a path to a local
            model directory, a Hugging Face model identifier, or an instance of
            PreTrainedModel.
        :return: The resolved PreTrainedModel instance for the verifier.
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
        verifier: Union[str, os.PathLike, PreTrainedModel],
        mode: Optional[Literal["full", "train_only"]] = None,
        add_to_config: bool = True,
    ) -> PreTrainedModel:
        """
        Attach a verifier model for the speculator that is used to attach to
        for running inference/training with the speculator and validates the
        candidate tokens generated by the speculator during the
        speculative decoding process. It should be compatible
        with the speculator's configuration in terms of vocabulary, architecture,
        and tokenization.

        Example:
            ```python
            # Load and attach a verifier
            verifier = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
            speculator.attach_verifier(verifier)

            # Now ready for generation
            outputs = speculator.generate(input_ids)
            ```

        :param verifier: The verifier model to attach. This can be a path to a local
            model directory, a Hugging Face model identifier, or an instance of
            PreTrainedModel. If a path or identifier is provided, the model will be
            loaded automatically. If an instance is provided, it will be used directly.
        :param mode: Optional mode for how the verifier is attached to the speculator.
            If "full", the verifier is fully integrated into the speculator's forward
            pass and generation methods. If "train_only", only the portions of the
            verifier needed for training are attached, allowing for better resource
            utilization during training. If None, defaults to "full".
        :param add_to_config: Whether to add the verifier that is being attached
            to the speculator's configuration. If True (default),
            the required references will be added to the speculator's config under
            `speculators_config.verifier`.
            If False, the speculator's configuration will not be modified,
        :return: The PreTrainedModel instance for the verifier that was attached.
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
            self.config.speculators_config.verifier = VerifierConfig.from_pretrained(
                verifier
            )

        return verifier

    def detach_verifier(self):
        """
        Removes the reference to the attached verifier model and frees up the
        associated memory. After calling this method, the speculator will not
        be able to perform forward passes or generation until a new verifier
        is attached.
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
        Overrides the state_dict method from PyTorch to ensure that save pathways
        within Transformers PreTrainedModel do not include the verifier model's
        parameters. This is important to ensure that the speculator model
        can be saved and loaded without including the verifier's state, which
        is expected to be managed separately.

        :param destination: Optional dictionary to store the state.
        :param prefix: Optional prefix for parameter names.
        :param keep_vars: Whether to keep Variables in the state_dict.
        :return: A dictionary containing the state of the speculator model,
            excluding the verifier model's parameters. This dictionary can be used
            to save the model's state to disk or for further processing.
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
        Defines the forward pass computation for the speculator model.

        This method must be implemented by all concrete speculator model
        subclasses. It defines how the model processes inputs to generate candidate
        tokens or logits specifically for training pipelines.

        Use `model.generate` for generation tasks, which will handle
        speculative decoding with the attached verifier.

        :param args: Positional arguments for the forward pass, typically including
            input_ids and potentially attention_mask, position_ids, etc.
        :param kwargs: Keyword arguments for the forward pass, which may include
            various model-specific parameters and options.
        :return: Model outputs, typically including logits or candidate token
            sequences, depending on the specific speculator implementation.
        """
        raise NotImplementedError(
            "The forward method is only supported on concrete "
            "speculator model subclasses."
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,  # noqa: ARG002
        generation_config: Optional[GenerationConfig] = None,  # noqa: ARG002
        logits_processor: Optional[LogitsProcessorList] = None,  # noqa: ARG002
        stopping_criteria: Optional[StoppingCriteriaList] = None,  # noqa: ARG002
        prefix_allowed_tokens_fn: Optional[  # noqa: ARG002
            Callable[[int, torch.Tensor], list[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,  # noqa: ARG002
        assistant_model: Optional["PreTrainedModel"] = None,  # type: ignore[override]  # noqa: ARG002
        streamer: Optional["BaseStreamer"] = None,  # noqa: ARG002
        negative_prompt_ids: Optional[torch.Tensor] = None,  # noqa: ARG002
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
        use_model_defaults: Optional[bool] = None,  # noqa: ARG002
        custom_generate: Optional[str] = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text using speculative decoding with the attached verifier model.
        The method follows the standard transformers generation interface, making it
        compatible with existing generation workflows while adding speculative
        decoding capabilities allowing for faster generation.

        :param inputs: The input token IDs to generate from. Can be None if input_ids
            are provided in kwargs.
        :param generation_config: Configuration for generation parameters like
            max_length, temperature, top_p, etc. If None, uses model defaults.
        :param logits_processor: List of logits processors to apply during generation
            for tasks like repetition penalty, top-k filtering, etc.
        :param stopping_criteria: List of stopping criteria to determine when to
            stop generation (e.g., max length, end-of-sequence tokens).
        :param prefix_allowed_tokens_fn: Function to constrain generation to allowed
            tokens based on the current prefix. Useful for structured generation.
        :param synced_gpus: Whether to synchronize GPUs during distributed generation.
            Relevant for multi-GPU setups.
        :param assistant_model: An assistant model to use for generation. This
            parameter maintains compatibility with transformers but may not be
            used in speculative decoding.
        :param streamer: A streamer to output tokens as they are generated, enabling
            real-time streaming of the generation process.
        :param negative_prompt_ids: Token IDs for negative prompting to steer
            generation away from certain content.
        :param negative_prompt_attention_mask: Attention mask for negative prompt
            tokens to properly handle padding.
        :param use_model_defaults: Whether to use model-specific default generation
            parameters instead of transformers defaults.
        :param kwargs: Additional keyword arguments for generation, including
            input_ids, attention_mask, max_length, etc.
        :return: Generated token sequences as either a GenerateOutput object
            (containing additional metadata) or a LongTensor of token IDs.
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
    Triggers the automatic discovery and registration of all
    SpeculatorModel subclasses found in the speculators.models package
    that have been registered with `SpeculatorModel.register(NAME)`. This
    enables dynamic model loading and instantiation based on configuration
    types without requiring explicit imports.
    """
    SpeculatorModel.auto_populate_registry()
