from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "DelsSpecSpeculatorConfig",
]


@SpeculatorModelConfig.register("dels_spec")
class DelsSpecSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for DeLS-Spec: Decoupled Long-Short Contexts speculator.

    DeLS-Spec adds a lightweight GRU-based local head (short-context expert)
    that captures intra-block causal dependencies missed by DFlash's parallel
    predictions. It trains independently with NTP loss.

    :param transformer_layer_config: Verifier config (for embedding dim / vocab)
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    :param block_size: Draft block size
    :param gru_hidden_size: Hidden size of the GRU
    :param low_rank_dim: Low-rank projection dimension before vocab head
    :param head_type: Local head variant ("rnn" or "markov")
    :param fusion_alpha: Weight for local head logits in product-of-experts fusion
    :param fusion_beta: Weight for unigram prior subtraction in fusion
    """

    speculators_model_type: Literal["dels_spec"] = "dels_spec"
    architectures: list[str] = Field(
        default_factory=lambda: ["DelsSpecSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Verifier config (used for embedding dim and vocab size)",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=16,
        description="Size of the draft block predicted per anchor",
    )

    gru_hidden_size: int = Field(
        default=1024,
        description="Hidden size of the GRU local head",
    )

    low_rank_dim: int = Field(
        default=256,
        description="Low-rank projection dimension before vocabulary head",
    )

    head_type: Literal["rnn", "markov"] = Field(
        default="rnn",
        description='Local head variant: "rnn" (GRU) or "markov" (bigram lookup)',
    )

    fusion_alpha: float = Field(
        default=0.3,
        description="Weight for local head logits in product-of-experts fusion",
    )

    fusion_beta: float = Field(
        default=0.3,
        description="Weight for unigram prior subtraction in fusion",
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking (needed for DFlash anchor selection)",
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
