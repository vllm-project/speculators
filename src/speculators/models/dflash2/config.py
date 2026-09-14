from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlash2SpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash2")
class DFlash2SpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash configuration with local convolutions and a candidate selector."""

    speculators_model_type: Literal["dflash2"] = "dflash2"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlash2DraftModel"],
        description="Model architectures that can load these weights",
    )
    sliding_window_non_causal: bool = Field(
        default=True,
        description="Use bidirectional masking inside sliding-window draft blocks.",
    )
    conv_kernel_size: int = Field(
        default=2,
        ge=1,
        description="Number of causal taps in each local dynamic convolution.",
    )
    conv_group_size: int = Field(
        default=16,
        ge=1,
        description="Number of hidden channels sharing each dynamic kernel.",
    )
    selector_rank: int = Field(
        default=256,
        ge=1,
        description="Rank of the candidate selector factorization.",
    )
    selector_top_k: int = Field(
        default=16,
        ge=1,
        description="Number of unary candidates reranked during inference.",
    )
