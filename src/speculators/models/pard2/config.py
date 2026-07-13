from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "Pard2SpeculatorConfig",
]


@SpeculatorModelConfig.register("pard2")
class Pard2SpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config with PARD-2 confidence-adaptive token (CAT) optimization.

    Replaces DFlash's fixed position decay with confidence-adaptive weights
    derived from the target model's per-token probability.  Adds a knowledge
    distillation loss alongside the base loss and supports dual-mode
    (target-dependent / target-independent) via stochastic feature dropout.
    """

    speculators_model_type: Literal["pard2"] = "pard2"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["Pard2Speculator"],
        description="Model architectures that can load these weights",
    )

    ce_alpha: float = Field(
        default=0.1,
        description=(
            "Weight of the hard cross-entropy loss term in the combined "
            "CE + KD training objective."
        ),
    )

    kd_alpha: float = Field(
        default=1.0,
        description="Weight of the knowledge-distillation (KL) loss term.",
    )

    kd_temperature: float = Field(
        default=1.0,
        description="Temperature applied to teacher and student logits in the KD term.",
    )

    target_feat_dropout: float = Field(
        default=0.1,
        description=(
            "Probability of dropping target features during training "
            "(Bernoulli gating for dual-mode support).  0 = always inject "
            "target features; 1 = never inject (target-independent only)."
        ),
    )
