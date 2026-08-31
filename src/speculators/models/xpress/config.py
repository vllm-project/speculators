from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "XPressSpeculatorConfig",
]


@SpeculatorModelConfig.register("xpress")
class XPressSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus the XPress causal-refiner head.

    The refiner adds a per-position logit bias built from the previous block
    token, the drafter's per-position hidden state, and a block-global hidden
    summary, mixed causally across the block. At inference the block is
    resolved with K parallel Jacobi passes instead of a serial per-position
    loop. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["xpress"] = "xpress"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["XPressSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "Block layout convention. False (XPress default): fill-in layout — "
            "slot 0 is the anchor, slots 1..B-1 predict; matches the released "
            "XPress checkpoints. True: DeepSeek/DSpark convention where every "
            "slot predicts the next token."
        ),
    )

    xpress_rank: int = Field(
        default=256,
        description="Low-rank dimension r of the refiner (embed, fuse, mixer, MLP "
        "all live in r-space).",
    )
    xpress_mlp_ratio: int = Field(
        default=2,
        description="Refiner MLP expansion: hidden = r * ratio (SwiGLU). The released\n"
        "XPress checkpoints use 2 (r=256 -> hidden 512); 4 doubles the head's MLP\n"
        "and is a different architecture.",
    )
    num_jacobi_passes: int = Field(
        default=6,
        description="Inference-time parallel refine passes K (engine-side knob; "
        "stored so converted checkpoints carry the validated default).",
    )

    anchor_cap_to_max_valid: bool = Field(
        default=False,
        description=(
            "Anchor sampling: cap the sampled anchors at "
            "min(max_anchors, valid_candidates - 1) rather than at "
            "valid_candidates. Worth ~0.3% of supervised blocks."
        ),
    )

    prefix_valid_mask: bool = Field(
        default=False,
        description=(
            "Loss mask: make the per-block validity mask a PREFIX (cumprod) "
            "mask -- once a slot is invalid every later slot in the block is "
            "masked too, because a left-to-right accept can never reach it."
        ),
    )

    eval_jacobi_passes: int | None = Field(
        default=None,
        description=(
            "TRAINING-time accept-length rollout passes. None = block_size - 1, "
            "which is what the in-training accept-length rollout uses; "
            "num_jacobi_passes (6) is its OFFLINE package-eval setting and is "
            "exported with the checkpoint, so the two must stay separate."
        ),
    )
