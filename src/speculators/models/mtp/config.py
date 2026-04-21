"""Configuration for MTP speculator model."""

from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModelConfig

__all__ = ["MTPConfig"]


@SpeculatorModelConfig.register("mtp")
class MTPConfig(SpeculatorModelConfig):
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

    step_weight_beta: float = Field(
        default=0.6,
        description=(
            "Exponential decay factor for per-step loss weights. "
            "Weight for step k is beta^(k-1) normalized over all steps. "
            "From FastMTP (arXiv:2509.18362), Equation 2."
        ),
    )

    @property
    def hidden_size(self) -> int:
        return self.transformer_layer_config.hidden_size  # type: ignore[return-value]

    @property
    def vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size  # type: ignore[return-value]

    @property
    def num_speculative_steps(self) -> int:
        return self.speculators_config.proposal_methods[0].speculative_tokens  # type: ignore[union-attr,attr-defined]

    @staticmethod
    def compute_step_weights(beta: float = 0.6, num_steps: int = 3) -> list[float]:
        """Compute normalized exponential-decay step weights.

        alpha_k = beta^(k-1) / sum(beta^(j-1) for j=1..K)

        See FastMTP (arXiv:2509.18362), Equation 2.
        """
        raw = [beta**k for k in range(num_steps)]
        total = sum(raw)
        return [w / total for w in raw]

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
