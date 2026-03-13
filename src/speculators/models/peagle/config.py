from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.eagle3.config import Eagle3SpeculatorConfig

__all__ = [
    "PEagleSpeculatorConfig",
]


@SpeculatorModelConfig.register("peagle")
class PEagleSpeculatorConfig(Eagle3SpeculatorConfig):
    """
    Configuration for P-EAGLE (Parallel EAGLE) speculator.

    P-EAGLE extends EAGLE-3 with parallel multi-token prediction using
    Conditional Drop Token (COD) sampling for memory-efficient training.

    :param para_depths: Number of parallel prediction groups (typically 8)
    :param down_sample_ratio: Geometric decay ratio for COD sampling (r in [0,1])
    :param down_sample_ratio_min: Minimum retention ratio floor
    :param ptd_token_id: Token ID for predicted token dropout (padding unused positions)
    :param max_seq_len: Maximum sequence length for attention mask construction
    """

    speculators_model_type: Literal["peagle"] = "peagle"
    architectures: list[str] = Field(
        default_factory=lambda: ["PEagleSpeculator"],
        description="Model architectures that can load these weights",
    )

    para_depths: int = Field(
        default=8,
        description="Number of parallel prediction groups (depths)",
        ge=1,
        le=16,
    )

    down_sample_ratio: float = Field(
        default=0.7,
        description="Geometric decay ratio for COD sampling (retention rate r)",
        gt=0.0,
        lt=1.0,
    )

    down_sample_ratio_min: float = Field(
        default=0.1,
        description="Minimum retention ratio floor to prevent over-sampling",
        gt=0.0,
        le=1.0,
    )

    ptd_token_id: int = Field(
        default=0,
        description="Token ID used for padding unused positions in parallel groups",
    )

    max_seq_len: int = Field(
        default=2048,
        description="Maximum sequence length for attention mask construction",
        ge=128,
        le=8192,
    )

    # Override Eagle3 default: P-EAGLE requires trainable embeddings
    # (matches p-eagle-train)
    embed_requires_grad: bool = Field(
        default=True,
        description=(
            "Whether embedding layer weights require gradients during "
            "training (True for P-EAGLE)"
        ),
    )

    prediction_loss_weight: float = Field(
        default=1.0,
        description="Weight for prediction loss (cross-entropy on logits). "
        "P-eagle-train uses only prediction loss, no hidden state distillation.",
        gt=0.0,
    )
