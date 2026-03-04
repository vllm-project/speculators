"""
Configuration for P-EAGLE (Parallel EAGLE) speculator model.

P-EAGLE extends EAGLE-3 with multi-token prediction through Conditional-On-Distribution
(COD) sampling, enabling 2-3x speedups over sequential EAGLE-3 drafting.

This configuration adds P-EAGLE-specific parameters on top of the EAGLE-3 config:
- para_num: Number of parallel prediction depths
- down_sample_ratio: COD geometric decay retention rate
- down_sample_ratio_min: Minimum retention rate
- ptd_token_id: Padding token ID for MTP positions
"""

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
    Configuration for P-EAGLE speculator with parallel multi-token prediction.

    P-EAGLE extends EAGLE-3 with parallel prediction groups generated via
    COD (Conditional-On-Distribution) sampling. Each depth k predicts the
    token k+1 positions ahead, all processed in a single forward pass.

    :param para_num: Number of parallel prediction depths (K)
    :param down_sample_ratio: COD retention rate r for geometric decay
    :param down_sample_ratio_min: Minimum retention rate at deepest depths
    :param ptd_token_id: Token ID used for MTP padding positions
    :param loss_type: Loss function type ('kl_div' or 'cross_entropy')
    """

    speculators_model_type: Literal["peagle"] = "peagle"
    architectures: list[str] = Field(
        default_factory=lambda: ["PEagleSpeculator"],
        description="Model architectures that can load these weights",
    )

    para_num: int = Field(
        default=8,
        ge=1,
        description=(
            "Number of parallel prediction depths (K). "
            "Depth k predicts the token k+1 positions ahead. "
            "All depths are processed in a single forward pass."
        ),
    )

    down_sample_ratio: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "COD retention rate r ∈ (0, 1) for geometric decay. "
            "Depth k retains n_valid × r^k positions from the training sequence. "
            "Lower values reduce memory but may hurt prediction quality at deeper depths."
        ),
    )

    down_sample_ratio_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum retention ratio to ensure at least some positions are kept "
            "at deeper prediction depths. Defaults to 0.0 (no minimum)."
        ),
    )

    ptd_token_id: int = Field(
        default=0,
        ge=0,
        description=(
            "Token ID used as padding for MTP positions that lack predicted tokens "
            "from previous steps. Renamed from 'unused_token_id' in the original paper "
            "for clarity (aligned with vLLM convention)."
        ),
    )

    loss_type: Literal["kl_div", "cross_entropy"] = Field(
        default="cross_entropy",
        description=(
            "Loss function type for training. P-EAGLE uses cross-entropy loss "
            "by default, unlike EAGLE-3 which uses KL divergence."
        ),
    )
