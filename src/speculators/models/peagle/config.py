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

    :param mask_token_id: Token ID used for masking
    """

    speculators_model_type: Literal["peagle"] = "peagle"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["PEagleSpeculator"],
        description="Model architectures that can load these weights",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for padding unused positions in parallel groups",
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
