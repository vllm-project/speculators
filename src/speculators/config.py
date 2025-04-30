import json
from pathlib import Path
from typing import Any, Optional, Union

from packaging.version import Version
from pydantic import BaseModel, Field
from transformers.utils import PushToHubMixin

__all__ = [
    "SpeculatorConfig",
    "DraftModelConfig",
    "TokenProposalConfig",
    "VerifierConfig",
    "__version__",
]

__version__ = "0.1.0"


class DraftModelConfig(BaseModel):
    """
    Configuration class for speculator draft architectures.
    It is used to define the architecture of the speculative draft model enabling
    various current and future techniques and architectures, including
    independent drafting (a separate model) and
    self-drafting (extensions of the verifier model).
    """

    type_: str = Field(
        description=(
            "The type of the speculator architecture. "
            "This must be a valid architecture name from the speculators library."
        )
    )
    inputs: list[str] = Field(
        description=(
            "List of input features for the speculator architecture. "
            "These can include features such as input_ids, "
            "specific intermediate states, the output of the verifier model, etc."
        )
    )
    model_config: dict[str, Any] = Field(
        description=(
            "Additional configuration settings and arguments for creating and "
            "configuring the draft model architecture. "
        )
    )


class TokenProposalConfig(BaseModel):
    """
    Configuration class for how the speculator generates and validates the draft tokens.
    It is used to define the type of token proposal algorithm to use and the supporting
    arguments / hyperparameters for a specific token proposal implementation including
    greedy, nucleus, token tree sampling, etc.
    """

    type_: str = Field(
        description=(
            "The type of the token proposal algorithm. "
            "This must be a valid proposer from the speculators library."
        )
    )
    args: dict[str, Any] = Field(
        description=(
            "Additional arguments for the verification criterion. "
            "These can include hyperparameters, configuration settings, etc."
        )
    )


class VerifierConfig(BaseModel):
    """
    Base class for the verifier model tied to the drafter for the speculator.
    It is used to define the various properties of the verifier model,
    enabling checks that a speculator is loaded with the correct model as well as
    storing additional info about the verifier model that can be used for constructing
    the drafter architecture and enabling complicated verification criteria.
    """

    architecture: str = Field(
        description=(
            "The architecture/class of the verifier model as defined in the "
            "transformers library. Ex: `LlamaPreTrainedModel`"
        )
    )
    model: str = Field(
        description=(
            "The name of the verifier model. This must be a valid HF id/name or "
            "a path to a local model directory. Ex: `meta-llama/Llama-3.3-70B-Instruct`"
        )
    )


class SpeculatorConfig(BaseModel, PushToHubMixin):
    """
    Configuration class for a speculator implementation enabling a standard interface
    for creating, loading, and saving algorithms for speculative decoding.
    """

    @staticmethod
    def from_pretrained(path: Union[str, Path]) -> "SpeculatorConfig":
        """
        Load a speculator configuration from a valid model on the Hugging Face Hub,
        a directory containing a valid speculator config.json file, or a path to a
        speculator config.json file.

        :param path: The path to the directory or config file.
        :return: An instance of SpeculatorConfig.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            # assume path is a Hugging Face Hub id/name for now
            # in the future, we'll check the Hub first, and if not found or error,
            # assume it's a local path following Transformers' logic.
            # for now, error out with NotImplementedError as a reminder to implement
            raise NotImplementedError(
                "Loading from the Hugging Face Hub is not yet implemented."
            )

        config_file = path / "config.json" if path.is_dir() else path

        with config_file.open("r", encoding="utf-8") as config_file_handle:
            config_data = json.load(config_file_handle)

        return SpeculatorConfig.model_validate(config_data)

    version: Version = Field(
        default=Version(__version__),
        description=(
            "The version of the speculator configuration class and file. "
            "This is updated on any major changes to the speculator class or "
            "file format, specifically ones that would break backward compatibility."
        ),
    )
    speculators_algorithm: str = Field(
        description=(
            "The name of the base algorithm or method used to create the speculator. "
            "This should be a valid algorithm name from the speculators library."
        )
    )
    draft_model: Union[str, DraftModelConfig] = Field(
        description=(
            "The draft model to use for the speculator. This can be a string "
            "representing the Hugging Face model id/name or a path to a local model "
            "directory, or a DraftModelConfig object defining the architecture and "
            "configuration settings for the draft model."
        ),
    )
    proposal_methods: dict[str, TokenProposalConfig] = Field(
        description=(
            "A dictionary of the supported token proposal methods for the speculator "
            "and their configurations, hyperparameters, and defaults. "
            "The keys are the method names, and the values are the configurations."
        )
    )
    default_proposal_method: str = Field(
        description=(
            "The default token proposal method to use for the speculator if not is "
            "provided or defaulted by the backend. This should be a valid key from "
            "the methods dictionary."
        )
    )
    verifier: Optional[VerifierConfig] = Field(
        description=(
            "The verifier model details and settings that the speculator is tied to / "
            "trained for. This includes details about the verifier model enabling "
            "checks that a speculator is loaded with the correct model as well as "
            "storing additional info about the verifier model that can be used for "
            "constructing the drafter architecture and enabling complicated "
            "verification criteria. If the speculator is not tied to a verifier model, "
            "this can be None."
        ),
    )

    def save(
        self,
        directory: Union[str, Path],
        push_to_hub: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> Path:
        """
        Save the speculator config to a config.json file in the specified directory.
        If push_to_hub is True, the config will be pushed to the Hugging Face Hub.

        :param directory: The directory to save the config file to.
        :param push_to_hub: Whether to push the config to the Hugging Face Hub.
        :param kwargs: Additional arguments to be used for the save operation.
        :return: The path to the saved config file.
        """
        if not isinstance(directory, Path):
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)
        config_file = directory / "config.json"
        with config_file.open("w", encoding="utf-8") as config_file_handle:
            config_file_handle.write(
                json.dumps(self.model_dump(), indent=4, ensure_ascii=False)
            )

        if push_to_hub:
            # Need to implement the logic to push to the Hugging Face Hub
            # For now, error out with NotImplementedError as a reminder to implement
            raise NotImplementedError(
                "Pushing to the Hugging Face Hub is not yet implemented."
            )

        return config_file
