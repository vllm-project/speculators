import os
from typing import Any, ClassVar, Literal, Optional, Union

from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from speculators import SpeculatorModelConfig, SpeculatorsConfig
from speculators.model import SpeculatorModel

__all__ = ["IndependentSpeculatorConfig"]


@SpeculatorModelConfig.register("independent")
class IndependentSpeculatorConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["independent"] = "independent"

    @classmethod
    def from_pretrained_config(
        cls, pretrained_config: PretrainedConfig, speculators_config: SpeculatorsConfig
    ) -> "IndependentSpeculatorConfig":
        pretrained_dict = pretrained_config.to_dict()
        pretrained_dict["model_type"] = pretrained_config.model_type

        return cls(**pretrained_dict, speculators_config=speculators_config)

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs
    ) -> "IndependentSpeculatorConfig":
        """
        Create a IndependentSpeculatorConfig from a dictionary, automatically
        instantiating the correct subclass based on the speculators_model_type field.

        :param config_dict: Dictionary containing the configuration
        :param kwargs: Additional keyword arguments that override config values
        :return: A IndependentSpeculatorConfig instance
        """
        dict_obj = {**config_dict, **kwargs}

        spec_model_type = dict_obj.setdefault("speculators_model_type", "independent")
        if spec_model_type != "independent":
            raise ValueError(
                f"Wrong speculators_model_type: {spec_model_type} for"
                "IndependentSpeculatorConfig."
            )

        if "model_type" not in dict_obj:
            raise ValueError("Expected model_type in config_dict")

        return cls.model_validate(dict_obj)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "IndependentSpeculatorConfig":
        """
        Load a IndependentSpeculatorConfig from the name/id of a model on the
        HuggingFace Hub or from a local directory.

        :param pretrained_model_name_or_path: The name or path to the pretrained model.
        :param cache_dir: The directory to cache the config in.
        :param force_download: Whether to force download the config from the Hub.
        :param local_files_only: Whether to use local files, not download from the Hub.
        :param token: The token to use for authentication with the Hub.
        :param revision: The revision of the config to load from the Hub.
        :param kwargs: Additional keyword arguments to pass to the config.
        :return: A IndependentSpeculatorConfig object with the loaded parameters.
        """
        # Transformers config loading
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )

        return cls.from_dict(config_dict, **kwargs)


@SpeculatorModel.register("independent")
class IndependentSpeculator(SpeculatorModel):
    config_class: ClassVar[type[IndependentSpeculatorConfig]] = (  # type: ignore[misc]
        IndependentSpeculatorConfig
    )

    _independent_speculator_mod_attributes = {
        "_draft_model",
        "_draft_model_class",
        "verifier",
        "verifier_attachment_mode",
    }

    def __init__(
        self,
        config: IndependentSpeculatorConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full", "train_only"]
        ] = None,
    ):
        if not isinstance(config, IndependentSpeculatorConfig):
            if not isinstance(config, PretrainedConfig):
                raise ValueError(
                    "Attempted to initialize a IndependentSpeculator with a"
                    f" {type(config)} class as the config class. Please use"
                    "a IndependentSpeculatorConfig instance or a subclass of"
                    "PretrainedConfig instead."
                )
            if (
                hasattr(config, "speculators_model_type")
                and config.speculators_model_type != "independent"
            ):
                raise ValueError(
                    "Attempted to initialize a IndependentSpeculator with a "
                    f"{config.speculators_model_type} config class. "
                    "IndependentSpeculator only supports models with "
                    "speculators_model_type='independent'."
                )
            # Subclass of PretrainedConfig but not an IndependentSpeculatorConfig
            # Convert to IndependentSpeculatorConfig
            config = IndependentSpeculatorConfig.from_pretrained_config(
                pretrained_config=config, speculators_config=None
            )

        self._draft_model = None

        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
        )

        config_class: type[PretrainedConfig] = CONFIG_MAPPING[config.model_type]
        self._draft_model_class: type[PreTrainedModel] = MODEL_FOR_CAUSAL_LM_MAPPING[  # type: ignore[assignment]
            config_class
        ]
        self._draft_model = self._draft_model_class(config)  # type: ignore[operator]

        self.post_init()

    def forward(self, *args, **kwargs):
        if self._draft_model is None:
            raise ValueError("Draft model is not initialized")

        return self._draft_model(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls: type["IndependentSpeculator"],
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
    ) -> "IndependentSpeculator":
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

        if isinstance(config, PretrainedConfig) and not isinstance(
            config, IndependentSpeculatorConfig
        ):
            # Convert PretrainedConfig to IndependentSpeculatorConfig
            config = IndependentSpeculatorConfig.from_dict(config.to_dict())

        if not isinstance(config, IndependentSpeculatorConfig):
            raise ValueError(
                f"Expected config to be an instance of IndependentSpeculatorConfig, "
                f"got {type(config)}."
            )

        if not pretrained_model_name_or_path and not kwargs.get("state_dict"):
            raise ValueError(
                "Either `pretrained_model_name_or_path` or `state_dict` must be "
                "provided to load a SpeculatorModel."
            )

        independent_speculator = cls(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
        )

        # Load the draft model
        independent_speculator._draft_model = (
            independent_speculator._draft_model_class.from_pretrained(
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
        )

        return independent_speculator

    def save_pretrained(self, *args, **kwargs):
        if self._draft_model is None:
            raise ValueError("Draft model is not initialized")
        self._draft_model.save_pretrained(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        if self._draft_model is None:
            raise ValueError("Draft model is not initialized")
        self._draft_model.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if self._draft_model is None:
            raise ValueError("Draft model is not initialized")
        return self._draft_model.state_dict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "_draft_model":
            return self._modules["_draft_model"]
        if self._draft_model is None:
            return super().__getattr__(name)

        if name in IndependentSpeculator._independent_speculator_mod_attributes:
            return super().__getattr__(name)

        return getattr(self._draft_model, name)

    def __setattr__(self, name: str, val: Any) -> None:
        # Allow patching over class attributes
        if hasattr(type(self), name):
            return super().__setattr__(name, val)

        if name in IndependentSpeculator._independent_speculator_mod_attributes:
            return super().__setattr__(name, val)

        if self._draft_model is None:
            return super().__setattr__(name, val)

        return setattr(self._draft_model, name, val)

    def __delattr__(self, name: str) -> None:
        # This mirrors `__setattr__`
        if hasattr(type(self), name):
            return super().__delattr__(name)

        if name in IndependentSpeculator._independent_speculator_mod_attributes:
            return super().__delattr__(name)

        if self._draft_model is None:
            return super().__delattr__(name)

        return delattr(self._draft_model, name)
