from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "DeLSSpecSpeculatorConfig",
]


@SpeculatorModelConfig.register("dels_spec")
class DeLSSpecSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for the DeLS-Spec local head (short-context expert).

    The local head is a lightweight GRU-based model that captures intra-block
    causal dependencies for parallel speculative drafting.  It trains
    independently of the DFlash backbone with a standard next-token prediction
    objective and fuses logits at inference via a product-of-experts formula.
    """

    speculators_model_type: Literal["dels_spec"] = "dels_spec"
    architectures: list[str] = Field(
        default_factory=lambda: ["DeLSSpecSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the target model (used for vocab/hidden sizes)",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=16,
        description="Block size for draft predictions (must match DFlash block_size)",
    )

    gru_hidden_size: int = Field(
        default=1024,
        description="Hidden dimension of the GRU in the RNN local head",
    )

    low_rank_dim: int = Field(
        default=256,
        description="Low-rank bottleneck dimension for the vocabulary projection",
    )

    head_type: Literal["rnn", "markov"] = Field(
        default="rnn",
        description=(
            "Local head variant: 'rnn' (1-layer GRU over block prefix) or "
            "'markov' (first-order bigram lookup)."
        ),
    )

    fusion_alpha: float = Field(
        default=0.3,
        description="Weight for the local-head logits in the product-of-experts fusion",
    )

    fusion_beta: float = Field(
        default=0.3,
        description="Weight for the unigram prior subtraction in fusion",
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
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
