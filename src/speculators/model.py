"""
Base model classes for the Speculators library.

This module contains the base model classes for speculative decoding implementations
in the Speculators library.
"""

import os
from abc import abstractmethod
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators.config import SpeculatorModelConfig
from speculators.utils import ClassRegistryMixin


class DraftVocabMixin(nn.Module):
    """
    Mixin for speculator models that use draft vocabulary mapping.

    Initializes vocab mapping buffers, token embeddings, and LM heads
    for models that implement draft-to-target vocabulary speculation.

    Requires the config to have ``transformer_layer_config`` and
    ``draft_vocab_size`` fields.
    """

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None
    embed_tokens: nn.Embedding
    lm_head: nn.Linear
    verifier_lm_head: nn.Linear
    verifier_norm: nn.Module

    def _init_vocab(self, config):
        """Initialize vocab mappings, token embeddings, and LM heads.

        Must be called after ``super().__init__(config=config)`` in the
        concrete model's ``__init__``.
        """
        # VOCAB MAPPINGS
        tl_config = config.transformer_layer_config
        self.draft_vocab_size = config.draft_vocab_size
        self.verifier_vocab_size = tl_config.vocab_size
        self.hidden_size = tl_config.hidden_size
        self.use_draft_vocab = self.draft_vocab_size != self.verifier_vocab_size
        t2d: torch.Tensor | None = None
        d2t: torch.Tensor | None = None
        if self.use_draft_vocab:
            t2d = torch.zeros((self.verifier_vocab_size,), dtype=torch.bool)
            d2t = torch.zeros((self.draft_vocab_size,), dtype=torch.long)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        # TOKEN EMBEDDINGS
        self.embed_tokens = nn.Embedding(
            self.verifier_vocab_size,
            self.hidden_size,
            padding_idx=getattr(tl_config, "pad_token_id", None),
        )
        self.embed_tokens.weight.requires_grad_(False)

        # LM HEADS
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
        self.verifier_lm_head = nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_lm_head.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False

        # Initialize weights to nan so it's easy to detect if they're never loaded
        torch.nn.init.constant_(self.lm_head.weight, torch.nan)
        torch.nn.init.constant_(self.embed_tokens.weight, torch.nan)
        torch.nn.init.constant_(self.verifier_lm_head.weight, torch.nan)
        self.lm_head._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001
        self.embed_tokens._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001
        self.verifier_lm_head._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001

    def load_vocab_mappings(self, t2d: torch.Tensor | None, d2t: torch.Tensor | None):
        """Load target-to-draft and draft-to-target vocabulary mapping tensors.

        Args:
            t2d: Target-to-draft vocabulary mapping tensor.
            d2t: Draft-to-target vocabulary mapping tensor.
        """
        if t2d is None and d2t is None:
            return
        elif t2d is None or d2t is None:
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be None. "
                f"Got t2d={'provided' if t2d is not None else 'None'}, "
                f"d2t={'provided' if d2t is not None else 'None'}"
            )

        if not self.use_draft_vocab:
            raise RuntimeError(
                "Vocab mappings (t2d/d2t) are not needed because "
                "draft_vocab_size equals verifier vocab_size. "
                "Set draft_vocab_size < verifier_vocab_size or "
                "omit t2d/d2t arguments."
            )

        if t2d.shape[0] != self.verifier_vocab_size:
            raise ValueError(
                f"t2d.shape[0] ({t2d.shape[0]}) must match"
                f" verifier_vocab_size ({self.verifier_vocab_size})."
            )
        if int(t2d.sum(dtype=torch.long).item()) != self.draft_vocab_size:
            raise ValueError(
                f"t2d has {int(t2d.sum(dtype=torch.long).item())} non-zero values, "
                f"expected {self.draft_vocab_size}."
            )

        if d2t.shape[0] != self.draft_vocab_size:
            raise ValueError(
                f"d2t.shape[0] ({d2t.shape[0]}) must match"
                f" draft_vocab_size ({self.draft_vocab_size})."
            )

        self.load_state_dict({"t2d": t2d, "d2t": d2t}, strict=False)

    def load_verifier_weights(self):
        """Load verifier model weights (embeddings, lm_head, etc.).

        Loads embed_tokens, lm_head, and verifier_lm_head weights from the
        verifier model. Handles draft vocab masking via t2d when use_draft_vocab
        is True. Subclasses can override to load additional weights (e.g. norms,
        tokenizer) by calling super().load_verifier_weights() first.
        """
        import warnings  # noqa: PLC0415

        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        speculators_config = getattr(
            getattr(self, "config", None), "speculators_config", None
        )
        if speculators_config is None:
            return
        verifier_config = speculators_config.verifier
        if verifier_config.name_or_path is None:
            return

        # Determine which weights to load based on model attributes
        weights_to_load = ["embed_tokens.weight", "lm_head.weight"]
        if hasattr(self, "verifier_norm"):
            weights_to_load.append("model.norm.weight")

        verifier_weights = load_model_layers(
            weights_to_load,
            verifier_config.name_or_path,
        )

        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        lm_head_weight = verifier_weights.get("lm_head.weight", embed_tokens_weight)

        # Load embed_tokens if not already loaded (NaN means uninitialized)
        if self.embed_tokens.weight.isnan().any():
            self.embed_tokens.load_state_dict({"weight": embed_tokens_weight})

        if self.use_draft_vocab:
            if self.t2d is None or not torch.any(self.t2d).item():  # type: ignore[arg-type]
                raise ValueError(
                    "t2d tensor hasn't been set. Please call "
                    "`.load_vocab_mappings(t2d, d2t)` before `.load_verifier_weights()`"
                )
            lm_head_weight = lm_head_weight[
                self.t2d.to(device=lm_head_weight.device, dtype=torch.bool), :  # type: ignore[union-attr,index]
            ]

        if self.lm_head.weight.isnan().any():
            self.lm_head.load_state_dict(
                {"weight": lm_head_weight.detach().clone()}, strict=False
            )
        self.verifier_lm_head.load_state_dict(
            {"weight": lm_head_weight.detach().clone()}, strict=False
        )

        # Load verifier norm weights if the model has verifier_norm
        if hasattr(self, "verifier_norm"):
            if "model.norm.weight" not in verifier_weights:
                warnings.warn(
                    f"Could not find final norm weights in "
                    f"{verifier_config.name_or_path}. "
                    "Using default initialization (weight=1.0).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                verifier_norm_sd = {"weight": verifier_weights["model.norm.weight"]}
                self.verifier_norm.load_state_dict(verifier_norm_sd)


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
        ```
    """

    # Registry configuration
    auto_package: ClassVar[str] = "speculators.models"
    registry_auto_discovery: ClassVar[bool] = True

    # PreTrainedModel settings
    config_class: ClassVar[type[SpeculatorModelConfig]] = SpeculatorModelConfig  # type: ignore[assignment,misc]
    base_model_prefix: ClassVar[str] = "model"  # type: ignore[misc]
    main_input_name: ClassVar[str] = "input_ids"  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = []  # type: ignore[assignment,misc]

    @classmethod
    def from_pretrained(
        cls: type["SpeculatorModel"],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
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
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                weights_only=weights_only,
                t2d=t2d,
                d2t=d2t,
                **kwargs,
            )

        model: SpeculatorModel = super().from_pretrained(  # type: ignore[misc]
            pretrained_model_name_or_path,
            *model_args,
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
        if hasattr(model, "load_vocab_mappings"):
            model.load_vocab_mappings(t2d, d2t)  # type: ignore[operator,attr-defined]
        if hasattr(model, "load_verifier_weights"):
            model.load_verifier_weights()  # type: ignore[operator,attr-defined]
        return model

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
    def verify_training_compatible(cls, model: "SpeculatorModel") -> None:
        """Verify that a model instance is compatible with training infrastructure.

        This method validates that the given model is:
        1. An instance of SpeculatorModel
        2. Registered in the SpeculatorModel registry
        3. Has a `layers` attribute (required for FSDP wrapping)

        Args:
            model: The model instance to verify

        Raises:
            TypeError: If model is not a SpeculatorModel instance
            ValueError: If model's class is not in the registry
            AttributeError: If model doesn't have a `layers` attribute
        """
        if not isinstance(model, SpeculatorModel):
            raise TypeError(
                f"Model must be a SpeculatorModel, got {type(model).__name__}"
            )

        model_class = type(model)
        registry = cls.registry
        if registry is None or model_class not in registry.values():
            raise ValueError(
                f"Model {model_class.__name__} is not registered in "
                f"SpeculatorModel.registry. "
                f"Available models: {list(registry.keys()) if registry else []}"
            )

        if not hasattr(model, "layers"):
            raise AttributeError(
                f"Model {model_class.__name__} must have a 'layers' attribute "
                f"containing decoder layers for FSDP wrapping"
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
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get algorithm-specific kwargs for training and validation.

        This method extracts algorithm-specific parameters from the training
        arguments and returns separate kwargs dictionaries for training and
        validation forward passes.

        Args:
            **kwargs: Training arguments containing algorithm-specific parameters.

        Returns:
            Tuple of (train_kwargs, val_kwargs) where:
                - train_kwargs: Dict passed to model.forward() during training
                - val_kwargs: Dict passed to model.forward() during validation

        Example:
            ```python
            @staticmethod
            def get_trainer_kwargs(**kwargs):
                train_kwargs = {
                    "num_steps": kwargs["num_steps"],
                    "use_special_mode": True,
                }
                val_kwargs = {
                    "num_steps": kwargs["num_steps"],
                    "use_special_mode": False,
                }
                return train_kwargs, val_kwargs
            ```
        """
        raise NotImplementedError(
            "Model must implement get_trainer_kwargs() staticmethod "
            "to support training infrastructure."
        )

    def __init__(self, config: SpeculatorModelConfig, **kwargs):
        """
        Initialize a SpeculatorModel instance.

        :param config: The configuration for the speculator model. Must be a
            SpeculatorModelConfig instance containing model hyperparameters and
            speculative decoding settings.
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

        config.tie_word_embeddings = False
        super().__init__(config, **kwargs)
        self.config: SpeculatorModelConfig = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The forward method is only supported on concrete "
            "speculator model subclasses."
        )
