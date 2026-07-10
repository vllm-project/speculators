from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

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
        default_factory=Qwen3Config,
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

    norm_before_fc: bool = Field(
        default=False,
        description=(
            "Apply a single RMSNorm to the concatenated auxiliary hidden states "
            "before the FC projection (gpt-oss style)."
        ),
    )

    fc_norm: bool = Field(
        default=False,
        description=(
            "Apply per-layer RMSNorm to each auxiliary hidden state before "
            "concatenation and FC projection — i.e. "
            "concat(Norm(h_a), Norm(h_b), Norm(h_c)) instead of the single "
            "Norm(concat(h_a, h_b, h_c)) used by norm_before_fc."
        ),
    )

    norm_output: bool = Field(
        default=False,
        description=(
            "Feed post-norm hidden states back across TTT steps to stabilize "
            "magnitude drift across speculation depths (Eagle 3.1)."
        ),
    )

    embed_requires_grad: bool = Field(
        default=False,
        description="Whether embedding layer weights require gradients during training",
    )

    @model_validator(mode="after")
    def _check_norm_flags(self) -> "Eagle3SpeculatorConfig":
        if self.norm_before_fc and self.fc_norm:
            raise ValueError(
                "norm_before_fc and fc_norm are mutually exclusive — "
                "enable one or the other, not both."
            )
        return self

    @model_validator(mode="after")
    def _check_aux_layer_ids(self) -> "Eagle3SpeculatorConfig":
        """Reject an explicit zero-width auxiliary hidden-state projection."""
        layer_ids = self.eagle_aux_hidden_state_layer_ids
        if layer_ids is not None and not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be None to use the legacy "
                "three-layer default, or contain at least one auxiliary layer ID."
            )
        if layer_ids is not None and any(layer_id < 1 for layer_id in layer_ids):
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must contain positive layer IDs"
            )
        if layer_ids is not None and len(set(layer_ids)) != len(layer_ids):
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must not contain duplicates"
            )
        return self

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen3Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size
