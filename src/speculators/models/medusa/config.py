"""Configuration for Medusa speculator model."""

from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig

__all__ = ["MedusaSpeculatorConfig"]


@SpeculatorModelConfig.register("medusa")
class MedusaSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for Medusa speculator.

    Architecture: K independent MLP heads (ResidualBlock), each predicting
    a future token at a different position. Each head consists of
    ``num_hidden_layers`` linear+SiLU+residual blocks followed by an
    output projection to vocab.

    :param num_heads: Number of Medusa prediction heads.
    :param num_hidden_layers: Number of hidden linear layers per head.
    :param medusa_hidden_size: Hidden dimension (must match verifier).
    :param medusa_vocab_size: Full verifier vocabulary size.
    :param truncated_vocab_size: Optional smaller vocab for speed.
    :param original_lm_head: If True, all heads share the verifier's
        lm_head for output projection. Otherwise each head has its own.
    :param medusa_fc_bias: Whether linear layers in heads use bias.
    :param head_weight_decay: Exponential decay base for per-head loss
        weights (lambda_k = decay^k).
    """

    speculators_model_type: Literal["medusa"] = "medusa"
    architectures: list[str] = Field(
        default_factory=lambda: ["MedusaModel"],
        description="Model architectures that can load these weights",
    )

    num_heads: int = Field(
        default=5,
        description="Number of Medusa prediction heads",
    )
    num_hidden_layers: int = Field(
        default=1,
        description="Number of hidden linear layers per head",
    )
    medusa_hidden_size: int = Field(
        default=0,
        description="Hidden dimension (derived from verifier)",
    )
    medusa_vocab_size: int = Field(
        default=0,
        description="Full verifier vocabulary size",
    )
    truncated_vocab_size: int = Field(
        default=0,
        description="Truncated vocab size (0 = use full vocab)",
    )
    original_lm_head: bool = Field(
        default=True,
        description="Share verifier lm_head across all heads",
    )
    medusa_fc_bias: bool = Field(
        default=False,
        description="Use bias in head linear layers",
    )
    head_weight_decay: float = Field(
        default=0.8,
        description="Exponential decay for per-head loss weights",
    )

    @property
    def hidden_size(self) -> int:
        return self.medusa_hidden_size

    @property
    def vocab_size(self) -> int:
        return self.medusa_vocab_size

    @property
    def draft_vocab_size(self) -> int:
        if self.truncated_vocab_size > 0:
            return self.truncated_vocab_size
        return self.vocab_size

    @property
    def num_speculative_steps(self) -> int:
        return self.num_heads
