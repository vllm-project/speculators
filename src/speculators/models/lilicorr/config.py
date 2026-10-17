from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "LiLiCorrSpeculatorConfig",
]


@SpeculatorModelConfig.register("lilicorr")
class LiLiCorrSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus the LiLiCorr candidate-lattice head.

    Defaults are the head geometry of the paper's main model
    (arXiv:2608.20530): K=8, width 1024, 2 layers, 8 heads, cosine factors
    scaled by a fixed 8.0. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["lilicorr"] = "lilicorr"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["LiLiCorrSpeculator"],
        description="Model architectures that can load these weights",
    )

    lilicorr_candidate_topk: int = Field(
        default=8, description="Candidates kept per slot (power of two <= 16)."
    )
    lilicorr_hidden_size: int | None = Field(
        default=1024, description="Head width; None uses the draft hidden size."
    )
    lilicorr_factor_dim: int = Field(
        default=1024, description="Width of the out/in factor vectors."
    )
    lilicorr_num_layers: int = Field(default=2, description="Lattice layers.")
    lilicorr_num_heads: int = Field(default=8, description="Lattice attention heads.")
    lilicorr_mlp_ratio: float = Field(default=2.0, description="Lattice MLP ratio.")
    lilicorr_logit_scale: float = Field(
        default=8.0, description="Fixed scale applied to the cosine factors."
    )
    lilicorr_vector_eps: float = Field(
        default=1e-4, description="Epsilon for L2-normalizing the factor vectors."
    )
