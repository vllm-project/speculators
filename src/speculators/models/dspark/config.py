from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for DSpark speculator with anchor-based training.

    DSpark features anchor-based training with Markov head and optional
    confidence head for speculative decoding.
    """

    speculators_model_type: Literal["dspark"] = "dspark"
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
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

    block_size: int = Field(
        default=8,
        description="Default size of the draft block predicted with a forward pass",
    )

    num_anchors: int = Field(
        default=256,
        description="Number of anchor positions to sample per sample during training",
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the target model to extract hidden states from",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking draft positions",
    )

    # DSpark-specific: Markov head
    markov_rank: int = Field(
        default=0,
        description="Rank of the Markov head. 0 disables the Markov head.",
    )
    markov_head_type: str = Field(
        default="vanilla",
        description="Type of Markov head: vanilla, gated, or rnn",
    )

    # DSpark-specific: Confidence head
    enable_confidence_head: bool = Field(
        default=False,
        description="Whether to enable the confidence head for early-stop",
    )
    confidence_head_with_markov: bool = Field(
        default=False,
        description="Whether the confidence head uses Markov embeddings as input",
    )

    # DSpark-specific: Loss weights
    ce_loss_alpha: float = Field(
        default=1.0,
        description="Weight for cross-entropy loss",
    )
    l1_loss_alpha: float = Field(
        default=0.0,
        description="Weight for L1 distribution matching loss",
    )
    confidence_head_alpha: float = Field(
        default=0.0,
        description="Weight for confidence head calibration loss",
    )
    loss_decay_gamma: float | None = Field(
        default=None,
        description="Decay rate for position-wise loss weighting",
    )

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
