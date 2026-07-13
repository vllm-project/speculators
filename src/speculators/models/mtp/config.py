"""Configuration for MTP speculator model."""

from typing import Any, Literal

import pydantic
from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModelConfig

__all__ = ["MTPSpeculatorConfig"]


@SpeculatorModelConfig.register("mtp")
class MTPSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for MTP (Multi-Token Prediction) speculator.

    Architecture: a single MTP layer with attention and MLP, combining
    verifier hidden states with token embeddings via an explicit input
    projection. ``embed_tokens`` and ``lm_head`` share the verifier's
    full vocabulary.

    :param transformer_layer_config: Configuration for the underlying
        transformer architecture (e.g., ``Qwen2Config``). All architecture
        dimensions are derived from this config.
    :param num_nextn_predict_layers: Number of MTP prediction heads in
        the checkpoint. vLLM reads this field directly to instantiate the
        correct number of MTP head instances. Currently only ``1`` is
        supported.
    """

    speculators_model_type: Literal["mtp"] = "mtp"
    architectures: list[str] = Field(
        default_factory=lambda: ["MTPDraftModel"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen2Config,
        description=(
            "Underlying transformer architecture config "
            "(e.g., Qwen2Config, Qwen3NextConfig)"
        ),
    )

    num_nextn_predict_layers: int = Field(
        default=1,
        description=(
            "Number of MTP prediction heads in the checkpoint. vLLM "
            "reads this field to create the correct number of "
            "speculator head instances."
        ),
    )

    num_centroids: int | None = Field(
        default=None,
        description=(
            "Number of centroids for a centroid-masked multi-level LM head. "
            "If None, standard flat LM head is used."
        ),
    )

    centroid_intermediate_top_k: int = Field(
        default=32,
        description="Number of top centroids to select during multi-level LM head decoding.",
    )

    @pydantic.model_validator(mode="after")
    def validate_centroids(self) -> "MTPSpeculatorConfig":
        if self.num_centroids is not None:
            if self.num_centroids < 1:
                raise ValueError(f"num_centroids must be >= 1, got {self.num_centroids}")
            if self.vocab_size % self.num_centroids != 0:
                raise ValueError(f"vocab_size ({self.vocab_size}) must be divisible by num_centroids ({self.num_centroids})")
            if self.centroid_intermediate_top_k > self.num_centroids:
                raise ValueError(f"centroid_intermediate_top_k ({self.centroid_intermediate_top_k}) cannot exceed num_centroids ({self.num_centroids})")
        return self

    @property
    def hidden_size(self) -> int:
        return self.transformer_layer_config.hidden_size  # type: ignore[return-value]

    @property
    def vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size  # type: ignore[return-value]

    @property
    def draft_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def num_speculative_steps(self) -> int:
        if (
            self.speculators_config is None
            or not self.speculators_config.proposal_methods
        ):
            raise ValueError(
                "Cannot determine num_speculative_steps: "
                "speculators_config is missing or has no proposal_methods. "
                "Provide a SpeculatorsConfig with at least one proposal method."
            )
        return self.speculators_config.proposal_methods[0].speculative_tokens  # type: ignore[union-attr,attr-defined]

    @field_validator("num_nextn_predict_layers")
    @classmethod
    def validate_num_nextn_predict_layers(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"MTP currently supports 1 layer, got {value}.")
        return value

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen2Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
