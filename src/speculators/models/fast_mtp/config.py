"""Configuration for FastMTP speculator model."""

from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModelConfig

__all__ = ["FastMTPConfig"]


@SpeculatorModelConfig.register("mtp")
class FastMTPConfig(SpeculatorModelConfig):
    """Configuration for FastMTP (Multi-Token Prediction) speculator.

    FastMTP predicts multiple future tokens per forward pass using a single MTP layer
    with teacher-forced multi-step loss for training.

    Architecture: a single MTP layer with attention and MLP, combining verifier hidden
    states with token embeddings via an explicit input projection. embed_tokens and
    lm_head are shared with the verifier model.

    :param transformer_layer_config: Configuration for the transformer architecture
        (e.g., Qwen2Config). All architecture dimensions are derived from this.
    :param num_nextn_predict_layers: Number of MTP prediction heads in the checkpoint.
        vLLM reads this field directly to instantiate the correct number of MTP head
        instances. Currently only 1 is supported.
    """

    speculators_model_type: Literal["mtp"] = "mtp"
    architectures: list[str] = Field(
        default_factory=lambda: ["FastMTPSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen2Config,
        description="Configuration for the transformer architecture (e.g., Qwen2Config)",  # noqa: E501
    )

    num_nextn_predict_layers: int = Field(
        default=1,
        description=(
            "Number of MTP prediction heads in the checkpoint. vLLM reads this "
            "field to create the correct number of speculator head instances."
        ),
    )

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size, derived from transformer_layer_config."""
        return self.transformer_layer_config.hidden_size  # type: ignore[return-value]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size, derived from transformer_layer_config."""
        return self.transformer_layer_config.vocab_size  # type: ignore[return-value]

    @property
    def num_speculative_steps(self) -> int:
        """Number of teacher-forced prediction steps, from the proposal config."""
        return self.speculators_config.proposal_methods[0].speculative_tokens  # type: ignore[union-attr,attr-defined]

    @field_validator("num_nextn_predict_layers")
    @classmethod
    def validate_num_nextn_predict_layers(cls, value: int) -> int:
        """Reject configs that request more than one FastMTP layer."""
        if value != 1:
            raise ValueError(
                f"FastMTP currently only supports 1 layer, got {value}. "
                "Multi-layer support may be added in future versions."
            )
        return value

    @field_serializer("transformer_layer_config")
    def serialize_transformer_layer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer_layer_config to dict for JSON storage."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_layer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config from dict or PretrainedConfig."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen2Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
