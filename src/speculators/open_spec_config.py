"""Translate the Open Spec Config DFlash wire format to runtime configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFlashSpeculativeConfig(BaseModel):
    """Settings defined by Open Spec Config's standalone DFlash schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    method: Literal["dflash"]
    speculative_tokens: int = Field(ge=1)
    verifier: str | None = Field(default=None, min_length=1)
    target_layer_ids: list[int] = Field(min_length=1)
    target_layer_start_idx: Literal[0, 1]
    draft_vocab_size: int | None = Field(default=None, ge=1)
    sample_from_anchor: bool
    mask_token_id: int = Field(ge=0)
    sliding_window_non_causal: bool
    training_framework: str | None = Field(default=None, min_length=1)
    training_framework_version: str | None = Field(default=None, min_length=1)

    @field_validator("target_layer_ids")
    @classmethod
    def validate_layer_ids(cls, value: list[int]) -> list[int]:
        """Require distinct, nonnegative verifier layer indices."""
        if min(value) < 0 or len(set(value)) != len(value):
            raise ValueError("target_layer_ids must be distinct and nonnegative")
        return value


def open_spec_to_native(config: dict) -> dict:
    """Normalize a DFlash wire config without fetching its verifier."""
    if config.get("open_spec_config_version") != "0.0.0":
        raise ValueError("Unsupported Open Spec Config version")
    if config.get("architectures") != ["DflashDraftModel"]:
        raise ValueError("Expected Open Spec Config architecture DflashDraftModel")
    settings = DFlashSpeculativeConfig.model_validate(config["speculative_config"])
    for key, value in config["speculative_config"].items():
        if value is None:
            raise ValueError(f"speculative_config.{key} cannot be null")
    transformer = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "open_spec_config_version",
            "speculative_config",
            "architectures",
            "speculators_metadata",
            "target_hidden_size",
        }
    }
    metadata = config.get("speculators_metadata", {})
    return {
        **metadata,
        "speculators_model_type": "dflash",
        "architectures": ["DFlashSpeculator"],
        "transformer_layer_config": transformer,
        "draft_vocab_size": settings.draft_vocab_size or transformer["vocab_size"],
        "block_size": settings.speculative_tokens + (not settings.sample_from_anchor),
        "aux_hidden_state_layer_ids": [
            index + 1 - settings.target_layer_start_idx
            for index in settings.target_layer_ids
        ],
        "target_hidden_size": config.get("target_hidden_size"),
        "sample_from_anchor": settings.sample_from_anchor,
        "mask_token_id": settings.mask_token_id,
        "sliding_window_non_causal": settings.sliding_window_non_causal,
        "speculators_config": {
            "algorithm": "dflash",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": settings.speculative_tokens,
                }
            ],
            "default_proposal_method": "greedy",
            "verifier": {
                "name_or_path": settings.verifier,
                "architectures": metadata.get("verifier_architectures", []),
            },
        },
    }
