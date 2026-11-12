from typing import Literal, Self

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.dspine_settings import DSpineSettings
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSpineSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspine")
class DSpineSpeculatorConfig(DSpineSettings, DFlashSpeculatorConfig):
    """DFlash configuration with depth-wise adjacent causal injection."""

    speculators_model_type: Literal["dspine"] = "dspine"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSpineDraftModel"],
        description="Model architectures that can load these weights",
    )

    @model_validator(mode="after")
    def _require_causal_attention(self) -> Self:
        if getattr(self.transformer_layer_config, "is_causal", None) is False:
            raise ValueError("DSpine requires causal attention.")
        self.transformer_layer_config.is_causal = True
        return self
