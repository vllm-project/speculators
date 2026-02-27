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

    FastMTP predicts multiple future tokens per forward pass using a single layer
    with weighted multi-step loss for training.

    Architecture components:
    - Single MTP layer with attention and MLP
    - Input projection combines hidden states + token embeddings
    - Shared lm_head with verifier model
    - Multi-step prediction with step-wise loss weighting

    :param transformer_config: Configuration for transformer architecture
        (e.g., Qwen2Config)
    :param num_speculative_steps: Number of future tokens to predict per forward pass
    :param num_nextn_predict_layers: Number of MTP layers (currently only 1 supported)
    :param mtp_loss_step_weights: Loss weights for each prediction step
    :param hidden_size: Hidden dimension size
    :param intermediate_size: FFN intermediate size
    :param num_attention_heads: Number of attention heads
    :param num_key_value_heads: Number of key-value heads for grouped query attention
    :param vocab_size: Vocabulary size
    :param max_position_embeddings: Maximum sequence length
    :param rms_norm_eps: Epsilon for RMS normalization
    """

    speculators_model_type: Literal["mtp"] = "mtp"
    architectures: list[str] = Field(
        default_factory=lambda: ["FastMTPSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_config: PretrainedConfig = Field(
        default_factory=Qwen2Config,
        description=(
            "Configuration for the transformer architecture (e.g., Qwen2Config)"
        ),
    )

    num_speculative_steps: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of future tokens to predict per forward pass",
    )

    num_nextn_predict_layers: int = Field(
        default=1,
        description="Number of MTP layers (currently only 1 is supported)",
    )

    mtp_loss_step_weights: list[float] = Field(
        default=[0.51, 0.31, 0.18],
        description="Loss weights for each prediction step",
    )

    # Architecture parameters (for convenience, also stored in transformer_config)
    hidden_size: int = Field(
        default=4096,
        description="Hidden dimension size",
    )

    intermediate_size: int = Field(
        default=11008,
        description="FFN intermediate size",
    )

    num_attention_heads: int = Field(
        default=32,
        description="Number of attention heads",
    )

    num_key_value_heads: int = Field(
        default=8,
        description="Number of key-value heads for grouped query attention",
    )

    vocab_size: int = Field(
        default=151680,
        description="Vocabulary size",
    )

    max_position_embeddings: int = Field(
        default=32768,
        description="Maximum sequence length",
    )

    rms_norm_eps: float = Field(
        default=1e-6,
        description="Epsilon for RMS normalization",
    )

    @field_validator("num_nextn_predict_layers")
    @classmethod
    def validate_num_nextn_predict_layers(cls, value: int) -> int:
        """Validate that only 1 FastMTP layer is used.

        This implementation currently supports only a single FastMTP layer.
        Multi-layer support may be added in the future.

        :param value: Number of FastMTP layers
        :return: Validated value
        :raises ValueError: If value is not 1
        """
        if value != 1:
            raise ValueError(
                f"FastMTP currently only supports 1 layer, got {value}. "
                "Multi-layer support may be added in future versions."
            )
        return value

    @field_validator("mtp_loss_step_weights")
    @classmethod
    def validate_mtp_loss_step_weights(cls, value: list[float]) -> list[float]:
        """Validate FastMTP loss step weights.

        Weights should have length matching num_speculative_steps (validated in model),
        all be non-negative, and sum to a reasonable value (warning if far from 1.0).

        :param value: List of loss weights
        :return: Validated weights
        :raises ValueError: If weights are invalid
        """
        if not all(w >= 0 for w in value):
            raise ValueError(
                f"All FastMTP loss step weights must be non-negative, got {value}"
            )

        min_weight_sum = 0.1
        max_weight_sum = 10.0
        weight_sum = sum(value)
        if not (min_weight_sum <= weight_sum <= max_weight_sum):
            raise ValueError(
                f"Sum of FastMTP loss step weights should be between 0.1 and 10.0, "
                f"got {weight_sum:.2f}. Weights: {value}"
            )

        return value

    @field_serializer("transformer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer_config to dict for JSON storage.

        :param value: Transformer config object
        :return: Dictionary representation
        """
        return value.to_diff_dict()

    @field_validator("transformer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config.

        :param value: Either a PretrainedConfig or dict
        :return: PretrainedConfig object
        """
        if isinstance(value, dict):
            # Determine config class from model_type
            config_class: type[PretrainedConfig] = Qwen2Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
