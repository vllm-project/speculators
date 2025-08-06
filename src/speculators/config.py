"""
Configuration classes for Speculators library.

This module contains the configuration classes for speculative decoding
implementations in the Speculators library. These includes configurations for
token proposal methods, verifier models, speculative decoding algorithms,
and speculator models.

The configurations use Pydantic for validation, serialization, and deserialization,
and extend Hugging Face's PretrainedConfig where appropriate to maintain compatibility
with the transformers ecosystem.

Classes:
    TokenProposalConfig: Base configuration for token proposal methods
    VerifierConfig: Configuration for verifier models with compatibility validation
    SpeculatorsConfig: Configuration for speculative decoding implementations
    SpeculatorModelConfig: Configuration for speculator models with transformers
        compatibility
"""

import os
from importlib.metadata import version
from typing import Any, ClassVar, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from transformers import PretrainedConfig

from speculators.utils import PydanticClassRegistryMixin, ReloadableBaseModel

__all__ = [
    "SpeculatorModelConfig",
    "SpeculatorsConfig",
    "TokenProposalConfig",
    "VerifierConfig",
    "reload_and_populate_configs",
]


class TokenProposalConfig(PydanticClassRegistryMixin):
    """
    The base config for a token proposal method which defines how tokens are generated
    by the speculator, how they are passed to the verifier, and how they are scored
    for acceptance or rejection. All implementations of token proposal methods
    must inherit from this class, set the proposal_type to a unique value, and
    add any additional parameters needed to instantiate and implement the method.

    It uses pydantic to validate the parameters, provide default values, and
    enable automatic serialization and deserialization of the correct class
    types based on the proposal_type field.
    """

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type["TokenProposalConfig"]:
        if cls.__name__ == "TokenProposalConfig":
            return cls

        return TokenProposalConfig

    auto_package: ClassVar[str] = "speculators.proposals"
    registry_auto_discovery: ClassVar[bool] = True
    schema_discriminator: ClassVar[str] = "proposal_type"

    proposal_type: str = Field(
        description=(
            "The type of token proposal the config is for. "
            "Must be a supported proposal type from the Speculators repo."
        ),
    )


class VerifierConfig(BaseModel):
    """
    The base config for a verifier model which defines the parameters that are required
    to either load the original verifier model or validate compatibility with a new
    verifier based on the the architecture and tokenizers/processor properties.
    It provides convenience methods to extract the required parameters from a
    PretrainedConfig object.
    """

    @classmethod
    def from_config(
        cls, config: PretrainedConfig, name_or_path: Optional[str] = "UNSET"
    ) -> "VerifierConfig":
        """
        Create a VerifierConfig from a PretrainedConfig object.
        Used to extract the required parameters from the original verifier
        config and create a VerifierConfig object.

        :param config: The PretrainedConfig object to extract the parameters from.
        :param name_or_path: The name or path for the verifier model.
            Set to None to not add a specific name_or_path.
            If not provided, the name_or_path from the config will be used.
        :return: A VerifierConfig object with the extracted parameters.
        """
        config_dict = config.to_dict()

        if name_or_path == "UNSET":
            name_or_path = (
                getattr(config, "name_or_path", None)
                or config_dict.get("_name_or_path", None)
                or config_dict.get("name_or_path", None)
            )

        return cls(
            name_or_path=name_or_path,
            architectures=config_dict.get("architectures", []),
        )

    name_or_path: Optional[str] = Field(
        description=(
            "The name as a Hugging Face id or path to the verifier model "
            "used for the speculator. Used to dynamically load the verifier the "
            "speculator was created for."
        ),
    )
    architectures: list[str] = Field(
        description=(
            "The architectures for the original verifier as found in its config.json. "
            "Used to validate architecture compatibility of different verifiers "
            "with the speculator, if needed."
        ),
    )


class SpeculatorsConfig(ReloadableBaseModel):
    """
    The base config for a spec decode implementation which defines the parameters
    required to implement a speculators algorithm for the parent, speculator model.
    It includes details on the algorithm, token proposals, and the verifier model.
    """

    algorithm: str = Field(
        description=(
            "The speculative decoding algorithm the speculator implements. "
            "Must be an algorithm name from the Speculators library. "
        ),
    )
    proposal_methods: list[TokenProposalConfig] = Field(
        description=(
            "The token proposal methods supported by the speculator. "
            "Must be a list of supported proposal configs from the Speculators repo."
        ),
    )
    default_proposal_method: str = Field(
        description=(
            "The default token proposal method to use when no method is specified. "
            "Must be the proposal_type for one of items in the proposal_methods list."
        ),
    )
    verifier: VerifierConfig = Field(
        description=(
            "The config for the verifier the speculator was created for. "
            "Used to auto load the verifier when the speculator is loaded, if needed. "
            "Also used to validate the verifier architecture and tokenizer "
            "compatibility for a new verifier, if needed."
        ),
    )


class SpeculatorModelConfig(PydanticClassRegistryMixin, PretrainedConfig):
    """
    The base config for a speculator model and implementation which defines the
    hyperparameters and settings required to implement a speculator model.
    It includes details on the speculator model architecture along with the
    speculators config describing the algorithm, token proposals, and verifier model.

    It inherits from the Transformers PretrainedConfig class to ensure full
    compatibility with standard Transformers model pathways while building on
    the standard methods for PretrainedConfigs to load, save, and push to the HF hub.

    This is the main config which maps to the config.json file for saved speculators.
    """

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
    ) -> "SpeculatorModelConfig":
        """
        Load a SpeculatorModelConfig from the name/id of a model on the Hugging Face Hub
        or from a local directory. Will automatically instantiate the correct config
        from speculators.models package.

        :param pretrained_model_name_or_path: The name or path to the pretrained model.
        :param cache_dir: The directory to cache the config in.
        :param force_download: Whether to force download the config from the Hub.
        :param local_files_only: Whether to use local files, not download from the Hub.
        :param token: The token to use for authentication with the Hub.
        :param revision: The revision of the config to load from the Hub.
        :param kwargs: Additional keyword arguments to pass to the config.
        :return: A SpeculatorModelConfig object with the loaded parameters.
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

        if "speculators_model_type" not in config_dict:
            # Conversion pathway
            raise NotImplementedError(
                "Loading a non-speculator model config is not supported yet."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs
    ) -> "SpeculatorModelConfig":
        """
        Create a SpeculatorModelConfig from a dictionary, automatically instantiating
        the correct subclass based on the speculators_model_type field.

        :param config_dict: Dictionary containing the configuration
        :param kwargs: Additional keyword arguments that override config values
        :return: A SpeculatorModelConfig instance of the appropriate subclass
        """
        dict_obj = {**config_dict, **kwargs}

        if "speculators_model_type" not in dict_obj:
            raise ValueError(
                "The config dictionary must contain the 'speculators_model_type' field "
                "for loading a SpeculatorModelConfig in the Speculators library."
            )

        return cls.model_validate(dict_obj)

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type["SpeculatorModelConfig"]:
        if cls.__name__ == "SpeculatorModelConfig":
            return cls

        return SpeculatorModelConfig

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Registry configuration
    auto_package: ClassVar[str] = "speculators.models"
    registry_auto_discovery: ClassVar[bool] = True
    schema_discriminator: ClassVar[str] = "speculators_model_type"

    # PretrainedConfig class attributes
    base_config_key: ClassVar[str] = ""  # type: ignore[misc]
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {}  # type: ignore[misc,assignment]
    is_composition: ClassVar[bool] = False  # type: ignore[misc]
    attribute_map: ClassVar[dict[str, str]] = {}  # type: ignore[misc]
    base_model_tp_plan: ClassVar[Optional[dict[str, Any]]] = None  # type: ignore[misc]
    base_model_pp_plan: ClassVar[Optional[dict[str, tuple[list[str]]]]] = None  # type: ignore[misc]

    # Speculator model instance attributes
    speculators_model_type: str = Field(
        default="",
        description="The type of model from the Speculators repo this config is for.",
    )
    speculators_version: str = Field(
        default=version("speculators"),
        description="Version of the speculators library",
    )
    speculators_config: SpeculatorsConfig = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "The speculators config describing what the model implements and creation. "
            "Contains information about the algorithm, proposal methods, and verifier."
        ),
    )

    def __init__(self, **kwargs):
        # initialize the Pydantic arguments first to set all valid fields
        PydanticClassRegistryMixin.__init__(self, **kwargs)

        # Set model_type to speculator_model if not already set
        self.model_type = kwargs.setdefault("model_type", "speculator_model")

        # reset kwargs handled by Pydantic so PretrainedConfig doesn't override
        for field in self.__class__.model_fields:
            kwargs[field] = getattr(self, field)

        # initialize the Hugging Face PretrainedConfig arguments for the model
        PretrainedConfig.__init__(self, **kwargs)

        # ensure we always update the transformers version
        self.transformers_version = version("transformers")

    def to_dict(self) -> dict[str, Any]:
        """
        :return: A dictionary representation of the full config, including the
            PretrainedConfig variables and Pydantic model fields.
        """
        pretrained_dict = super().to_dict()
        model_dict = self.model_dump()
        config_dict = {**pretrained_dict, **model_dict}

        # strip all class variables and metadata that are not needed in the output
        for key in (
            "model_config",
            "auto_package",
            "registry_auto_discovery",
            "schema_discriminator",
            "base_config_key",
            "sub_configs",
            "is_composition",
            "attribute_map",
            "base_model_tp_plan",
            "base_model_pp_plan",
        ):
            config_dict.pop(key, None)

        return config_dict

    def to_diff_dict(self) -> dict[str, Any]:
        """
        :return: A dictionary representation of a simplified config,
            including only the PretrainedConfig fields that have been modified
            or set, along with all Pydantic fields.
        """
        return super().to_diff_dict()


def reload_and_populate_configs():
    """
    Automatically populates the registry for all PydanticClassRegistryMixin subclasses
    and reloads schemas for all Config classes to ensure their schemas are up-to-date
    with the current registry state.
    """
    TokenProposalConfig.auto_populate_registry()
    SpeculatorsConfig.reload_schema()
    SpeculatorModelConfig.auto_populate_registry()
