"""
Eagle model implementation for EAGLE1 and HASS speculator models.

This module provides a unified implementation for both EAGLE1 and HASS variants
through configurable parameters.

Classes:
    EagleSpeculatorConfig: Configuration for EAGLE/HASS models
"""

from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator
from transformers import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from typing_extensions import Self

from speculators.config import SpeculatorModelConfig

__all__ = [
    "EagleSpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle")
class EagleSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration class for EAGLE1 and HASS speculator models.

    This unified configuration supports both EAGLE1 and HASS variants through
    configurable parameters, allowing a single model implementation to handle
    both architectures.
    """

    speculators_model_type: Literal["eagle"] = "eagle"
    architectures: list[str] = Field(
        default_factory=lambda: ["EagleSpeculator"],
        description=(
            "List of model architectures that can be used with the "
            "model pretrained weights."
        ),
    )
    transformer_layer_architecture: str = Field(
        default="LlamaDecoderLayer",
        description=(
            "The architecture of the transformer layer to use. "
            "Typically 'LlamaDecoderLayer' for Eagle 1, Eagle 2, and HASS."
        ),
    )
    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description=(
            "Configuration for the transformer layer to use in the "
            "Eagle model architecture. This must be a PretrainedConfig that matches "
            "the config required by the transformer_layer_architecture."
        ),
    )
    layernorms: bool = Field(
        default=False,
        description=(
            "Whether to use additional layernorms in the model architecture, "
            "specifically the layernorm after the verifier's hidden state, "
            "after the fusion layer, and before the LM head. "
            "For Eagle, Eagle 1, and HASS, this is False."
        ),
    )
    fusion_bias: bool = Field(
        default=False,
        description=(
            "Whether to add a bias to the fusion (fc) layer that is applied to the "
            "concat of the input embeddings and input hidden state. "
            "For Eagle and Eagle 2, this is False, while for HASS it is True."
        ),
    )

    @model_validator(mode="after")
    def check_add_architectures(self) -> Self:
        """
        Ensure that the transformer_layer_architecture is included in the
        architectures list.

        :return: Self
        """
        if self.transformer_layer_architecture not in self.architectures:
            self.architectures.append(self.transformer_layer_architecture)

        return self

    @field_serializer("transformer_layer_config")
    def serialize_transformer_layer_config(self, value: PretrainedConfig) -> dict:
        """
        Serialize the transformer_layer_config to a dictionary.

        :param value: The PretrainedConfig instance to serialize.
        :return: Serialized dictionary representation of the config.
        """
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_layer_config(cls, value: Any) -> PretrainedConfig:
        """
        Validate that the transformer_layer_config is a valid PretrainedConfig.

        :param value: The instance to validate to a PretrainedConfig.
        :return: The validated PretrainedConfig instance.
        """
        if isinstance(value, dict):
            return PretrainedConfig.from_dict(value)

        if isinstance(value, PretrainedConfig):
            return value

        raise ValueError(
            "transformer_layer_config must be a PretrainedConfig or a dict "
            "that can be converted to a PretrainedConfig."
        )
