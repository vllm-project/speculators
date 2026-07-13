from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlareSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflare")
class DFlareSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config with adaptive layer fusion and heterogeneous KV projections.

    DFlare replaces DFlash's single FC-projected shared representation with
    per-layer learnable scalar fusion weights over target hidden states and
    adds separate KV projection matrices for target context vs draft states.
    """

    speculators_model_type: Literal["dflare"] = "dflare"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlareSpeculator"],
        description="Model architectures that can load these weights",
    )

    use_heterogeneous_kv: bool = Field(
        default=True,
        description="Use separate KV projections for target context vs draft states.",
    )

    progressive_gamma: bool = Field(
        default=True,
        description="Linearly warm up the loss decay gamma during training.",
    )

    gamma_start: float = Field(
        default=4.5,
        description="Initial gamma value for progressive decay schedule.",
    )

    gamma_max: float = Field(
        default=10.5,
        description="Final gamma value for progressive decay schedule.",
    )
