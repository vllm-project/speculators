from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DNeoSpeculatorConfig",
]


@SpeculatorModelConfig.register("dneo")
class DNeoSpeculatorConfig(DFlashSpeculatorConfig):
    """DNeo: a DFlash-family parallel drafter with structure-preserving feature
    encoding, prefix-only attention, 2-tap local convolutions and a fused causal
    head (arXiv:2611.21925).

    The drafter's hidden size must equal the target's: the feature encoder passes
    target coordinates through unchanged instead of projecting them.
    """

    speculators_model_type: Literal["dneo"] = "dneo"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DNeoDraftModel"],
        description="Model architectures that can load these weights",
    )

    feature_encoder: Literal["mofa", "diagonal"] = Field(
        default="mofa",
        description=(
            "How target hidden states are fused into the drafter's context features. "
            "'mofa': one softmax weight per target layer (Mixture-of-Features "
            "Attention). 'diagonal': one weight per target layer and hidden "
            "coordinate (the paper's restricted encoder). Neither mixes coordinates."
        ),
    )
    prefix_only_attention: bool = Field(
        default=True,
        description=(
            "Draft positions attend only to the context features, not to each other; "
            "the local convolutions and the causal head carry the in-block "
            "dependencies."
        ),
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
    causal_head_rank: int = Field(
        default=256,
        ge=0,
        description=(
            "Rank of the fused DNeo causal head (DSpark low-rank bias plus a "
            "DFlash-2 style context-aware term, combined in latent space). "
            "0 disables the head."
        ),
    )
