from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorModelConfig

__all__ = [
    "Eagle3SpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle3")
class Eagle3SpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for EAGLE-3 speculator with vocabulary mapping.

    EAGLE-3 features vocabulary mapping between draft (32K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    :param norm_before_residual: Apply hidden_norm before storing residual
    """

    speculators_model_type: Literal["eagle3"] = "eagle3"
    architectures: list[str] = Field(
        default_factory=lambda: ["Eagle3Speculator"],
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

    norm_before_residual: bool = Field(
        default=False,
        description="Apply hidden_norm before storing residual",
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    eagle_aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the Eagle auxiliary hidden state layers",
    )

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
