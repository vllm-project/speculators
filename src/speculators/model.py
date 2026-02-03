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
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal, Optional

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

from speculators.config import SpeculatorModelConfig
from speculators.utils import ClassRegistryMixin


class SpeculatorModel(ClassRegistryMixin, PreTrainedModel):  # type: ignore[misc]
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
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        verifier_attachment_mode: Literal["detached", "full", "train_only"]
        | None = None,
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
            or an instance of PreTrainedModel. The speculator will use this
            verifier for speculative decoding.
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

    @classmethod
    @abstractmethod
    def from_training_args(
        cls, verifier_config: PretrainedConfig, **kwargs
    ) -> "SpeculatorModel":
        """Create model instance from training arguments.

        This factory method is used by the training script to instantiate models
        from command-line arguments. Each algorithm must implement this to support
        the training infrastructure.

        Args:
            verifier_config: Configuration from the verifier/base model.
            **kwargs: Training arguments as keyword arguments. Each algorithm
                extracts the parameters it needs.

        Returns:
            Initialized model instance ready for training.

        Example:
            ```python
            @classmethod
            def from_training_args(cls, verifier_config, **kwargs):
                config = MySpeculatorConfig(
                    transformer_layer_config=verifier_config,
                    num_layers=kwargs['num_layers'],
                    ...
                )
                return cls(config=config, t2d=kwargs.get('t2d'), d2t=kwargs.get('d2t'))
            ```
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_training_args() classmethod "
            "to support training infrastructure."
        )

    @staticmethod
    @abstractmethod
    def get_trainer_kwargs(args) -> tuple[dict, dict]:
        """Get algorithm-specific kwargs for training and validation.

        This method extracts algorithm-specific parameters from the training arguments
        and returns separate kwargs dictionaries for training and validation forward passes.

        Args:
            args: Training arguments namespace containing algorithm-specific parameters.

        Returns:
            Tuple of (train_kwargs, val_kwargs) where:
                - train_kwargs: Dict passed to model.forward() during training
                - val_kwargs: Dict passed to model.forward() during validation

        Example:
            ```python
            @staticmethod
            def get_trainer_kwargs(args):
                train_kwargs = {
                    "num_steps": args.num_steps,
                    "use_special_mode": True,
                }
                val_kwargs = {
                    "num_steps": args.num_steps,
                    "use_special_mode": False,
                }
                return train_kwargs, val_kwargs
            ```
        """
        raise NotImplementedError(
            "Model must implement get_trainer_kwargs() staticmethod "
            "to support training infrastructure."
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
        storage and optional verifier model attachment. The verifier model is used
        during speculative decoding to validate the tokens proposed by the speculator.

        If no verifier is provided during initialization, it must be attached later
        using the attach_verifier method before calling generate.

        :param config: The configuration for the speculator model. Must be a
            SpeculatorModelConfig instance containing model hyperparameters and
            speculative decoding settings.
        :param verifier: The verifier model to attach. This can be a path to a local
            model directory, a Hugging Face model identifier, or an instance of
            PreTrainedModel. The speculator will use this verifier for
            speculative decoding. 
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
        self.verifier: PreTrainedModel | None = None
        self.verifier_attachment_mode: Literal["detached", "full", "train_only"] = (
            "detached"
        )

        verifier = verifier or config.speculators_config.verifier.name_or_path

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

def reload_and_populate_models():
    """
    Triggers the automatic discovery and registration of all
    SpeculatorModel subclasses found in the speculators.models package
    that have been registered with `SpeculatorModel.register(NAME)`. This
    enables dynamic model loading and instantiation based on configuration
    types without requiring explicit imports.
    """
    SpeculatorModel.auto_populate_registry()
