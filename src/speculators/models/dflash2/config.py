from typing import Literal

from pydantic import Field, model_validator

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
    draft_ffn_type: Literal["dense", "moe"] = Field(
        default="dense",
        description="Feed-forward implementation used by every draft layer.",
    )
    num_experts: int = Field(
        default=256,
        ge=1,
        description="Number of routed experts when draft_ffn_type='moe'.",
    )
    num_experts_per_tok: int = Field(
        default=8,
        ge=1,
        description="Number of routed experts selected per token.",
    )
    moe_intermediate_size: int = Field(
        default=512,
        ge=1,
        description="Intermediate width of each routed expert.",
    )
    shared_expert_intermediate_size: int = Field(
        default=512,
        ge=1,
        description="Intermediate width of the always-on shared expert.",
    )
    moe_experts_implementation: Literal[
        "grouped_mm", "batched_mm", "deepgemm", "sonicmoe", "reference"
    ] = Field(
        default="grouped_mm",
        description=(
            "Kernel used for the routed-expert GEMMs. This is a closed set on "
            "purpose: Transformers silently falls back to the per-expert Python "
            "loop when it does not recognize the name, and that loop costs about "
            "68x a grouped GEMM at 256 experts / hidden 2048 / intermediate 512 "
            "(measured on H200, ~505ms vs ~7.4ms per layer forward+backward). "
            "'reference' names that loop explicitly for debugging; 'batched_mm' "
            "materializes a dense per-expert activation and will exhaust device "
            "memory at production expert counts."
        ),
    )

    @model_validator(mode="after")
    def validate_moe_routing(self) -> "DFlash2SpeculatorConfig":
        if self.draft_ffn_type == "moe" and self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                "num_experts_per_tok cannot exceed num_experts: "
                f"{self.num_experts_per_tok} > {self.num_experts}."
            )
        return self
