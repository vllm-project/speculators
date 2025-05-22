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
    def from_verifier_config(
        cls, config: PretrainedConfig, name_or_path: Optional[str] = None
    ) -> "VerifierConfig":
        """
        Create a VerifierConfig from a PretrainedConfig object.
        Used to extract the required parameters from the original verifier
        config and create a VerifierConfig object.

        :param config: The PretrainedConfig object to extract the parameters from.
        :param name_or_path: The name or path for the verifier model.
            If not provided, the name_or_path from the config will be used.
        :return: A VerifierConfig object with the extracted parameters.
        """
        config_dict = config.to_dict()

        return cls(
            name_or_path=name_or_path or config.name_or_path,
            architectures=config_dict.get("architectures", []),
            hidden_size=config_dict.get("hidden_size", -1),
            intermediate_size=config_dict.get("intermediate_size", -1),
            vocab_size=config_dict.get("vocab_size", -1),
            max_position_embeddings=config_dict.get("max_position_embeddings", -1),
            bos_token_id=config_dict.get("bos_token_id", -1),
            eos_token_id=config_dict.get("eos_token_id", -1),
        )

    name_or_path: str = Field(
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
    hidden_size: int = Field(
        description=(
            "The hidden size of the original verifier as found in its config.json. "
            "Used to validate architecture compatibility of different verifiers "
            "with the speculator, if needed."
        ),
    )
    intermediate_size: int = Field(
        description=(
            "The intermediate size of original verifier as found in its config.json. "
            "Used to validate architecture compatibility of different verifiers "
            "with the speculator, if needed."
        ),
    )
    vocab_size: int = Field(
        description=(
            "The vocab size of the original verifier as found in the its config.json. "
            "Used to validate tokenizer compatibility of different verifiers "
            "with the speculator, if needed."
        ),
    )
    max_position_embeddings: int = Field(
        description=(
            "The max position embeddings of original verifier as in its config.json. "
            "Used to validate max length compatibility of different verifiers "
            "with the speculator, if needed."
        ),
    )
    bos_token_id: Union[int, list[int]] = Field(
        description=(
            "The beginning of sentence token id of the original verifier as "
            "found in its config.json. "
            "Used to validate tokenizer compatibility of different verifiers "
            "with the speculator, if needed."
        ),
    )
    eos_token_id: Union[int, list[int]] = Field(
        description=(
            "The end of sentence token id of the original verifier as "
            "found in its config.json. "
            "Used to validate tokenizer compatibility of different verifiers "
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


class SpeculatorModelConfig(BaseModel, PretrainedConfig):
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
        raise NotImplementedError("from_pretrained is not implemented yet.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Get around to_diff_dict constructor
    is_composition: ClassVar[bool] = True  # type: ignore[misc]

    speculators_model_type: str = Field(
        description=("The type of model from the Speculators repo this config is for.")
    )
    speculators_version: str = Field(
        default=version("speculators"),
        description="Version of the speculators library",
    )
    speculators_config: SpeculatorsConfig = Field(
        description=(
            "The speculators config describing what the model implements and creation. "
            "Contains information about the algorithm, proposal methods, and verifier."
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """
        :return: A dictionary representation of the full config, including the
            PretrainedConfig variables.
        """
        return self.model_dump()

    def to_diff_dict(self) -> dict[str, Any]:
        """
        :return: A dictionary representation of the config minus any base
            properties that are not needed for the diff. Uses the super's
            to_diff_dict method.
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
    SpeculatorModelConfig.reload_schema()
