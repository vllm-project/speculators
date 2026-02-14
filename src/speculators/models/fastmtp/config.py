from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorModelConfig

__all__ = [
    "FastMTPSpeculatorConfig",
]


@SpeculatorModelConfig.register("fastmtp")
class FastMTPSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for FastMTP speculator (arXiv:2509.18362).

    A single MTP head is applied recursively K times with position-shared weights
    to predict K future tokens for speculative decoding.
    """

    speculators_model_type: Literal["fastmtp"] = "fastmtp"
    architectures: list[str] = Field(
        default_factory=lambda: ["FastMTPDraftModel"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    num_speculative_steps: int = Field(
        default=3,
        description="Number of recursive prediction steps (K)",
    )

    @property
    def target_vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
