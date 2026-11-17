"""Serializable choices for the anchored-block model family."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

ObjectiveName = Literal["dflash", "dspark", "dflash2", "xpress"]
_CORRECTION_OBJECTIVES: dict[str, ObjectiveName] = {
    "identity": "dflash",
    "markov": "dspark",
    "selector": "dflash2",
    "xpress": "xpress",
}


class BackboneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["dflash", "dflash2"] = "dflash"
    conv_kernel_size: int = Field(default=2, ge=1)
    conv_group_size: int = Field(default=16, ge=1)


class CorrectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["identity", "markov", "selector", "xpress"] = "identity"
    rank: int = Field(default=256, ge=1)
    markov_head_type: Literal["vanilla", "gated", "rnn"] = "vanilla"
    selector_top_k: int = Field(default=16, ge=1)
    xpress_mlp_ratio: int = Field(default=2, ge=1)
    num_jacobi_passes: int = Field(default=6, ge=1)


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    with_correction_features: bool = False


class BlockRecipe(BaseModel):
    """One backbone, one correction, and an explicitly compatible objective."""

    model_config = ConfigDict(extra="forbid")

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    correction: CorrectionConfig = Field(default_factory=CorrectionConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    objective: ObjectiveName | None = None

    @model_validator(mode="after")
    def validate_objective(self):
        kind = self.correction.kind
        required = _CORRECTION_OBJECTIVES[kind]
        if self.confidence.enabled:
            if kind not in {"identity", "markov"}:
                raise ValueError(
                    "Confidence supervision requires identity or Markov correction"
                )
            required = "dspark"
        if self.confidence.with_correction_features and (
            not self.confidence.enabled or kind != "markov"
        ):
            raise ValueError(
                "Confidence features require an enabled head and Markov correction"
            )
        if self.objective is not None and self.objective != required:
            raise ValueError(
                f"Correction/confidence configuration requires objective={required!r}"
            )
        self.objective = required
        return self


@SpeculatorModelConfig.register("modular")
class ModularSpeculatorConfig(DFlashSpeculatorConfig):
    """Native block-model format; legacy serving export is a separate operation."""

    speculators_model_type: Literal["modular"] = "modular"  # type: ignore[assignment]
    has_no_defaults_at_init: ClassVar[bool] = True
    architectures: list[str] = Field(default_factory=lambda: ["ModularDraftModel"])
    checkpoint_format_version: Literal[1] | None = 1
    recipe: BlockRecipe = Field(default_factory=BlockRecipe)

    @model_validator(mode="after")
    def validate_components(self):
        backbone = self.recipe.backbone
        if self.block_size < 1:
            raise ValueError("Modular block_size must be positive")
        if not self.aux_hidden_state_layer_ids:
            raise ValueError(
                "Modular models require at least one target hidden-state layer"
            )
        if backbone.kind == "dflash2":
            if self.transformer_layer_config.hidden_size % backbone.conv_group_size:
                raise ValueError(
                    "Backbone hidden_size must be divisible by conv_group_size"
                )
            if backbone.conv_kernel_size > self.block_size:
                raise ValueError("Backbone conv_kernel_size cannot exceed block_size")
        if self.checkpoint_format_version != 1:
            raise ValueError(
                "Modular configurations require native checkpoint format version 1"
            )
        if (
            self.speculators_config is not None
            and self.speculators_config.algorithm != "modular"
        ):
            raise ValueError("Modular configurations must use algorithm='modular'")
        return self

    @model_validator(mode="after")
    def validate_dimensions(self):
        if (
            self.target_hidden_size is not None
            and self.target_hidden_size != self.transformer_layer_config.hidden_size
        ):
            raise ValueError(
                "Block backbones currently require matching verifier/draft hidden sizes"
            )
        correction = self.recipe.correction
        if correction.kind == "selector":
            if self.draft_vocab_size != self.transformer_layer_config.vocab_size:
                raise ValueError(
                    "Candidate selector requires the full verifier vocabulary"
                )
            if correction.selector_top_k > self.draft_vocab_size:
                raise ValueError("selector_top_k cannot exceed draft vocabulary size")
        return self
