from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from speculators import SpeculatorModelConfig
from speculators.open_spec_config import DFlashSpeculativeConfig

__all__ = [
    "DFlashSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash")
class DFlashSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for DFlash speculator with vocabulary mapping.

    DFlash features vocabulary mapping between draft (64K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    """

    speculators_model_type: Literal["dflash"] = "dflash"
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlashSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=8,
        description=(
            "Default size of the draft block predicted with a forward pass of the model"
        ),
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the DFlash auxiliary hidden state layers",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking",
    )

    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking within draft blocks for "
        "sliding window attention layers. Full attention layers are always "
        "bidirectional.",
    )

    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
        ),
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen3Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size

    def to_dict(self) -> dict[str, Any]:
        """Save DFlash using Open Spec Config; retain variant formats."""
        native = super().to_dict()
        if self.speculators_model_type != "dflash" or self.speculators_config is None:
            return native
        if self.aux_hidden_state_layer_ids is None or self.mask_token_id is None:
            raise ValueError(
                "Saving DFlash requires target layer IDs and mask_token_id"
            )
        settings = DFlashSpeculativeConfig(
            method="dflash",
            speculative_tokens=self.block_size - (not self.sample_from_anchor),
            verifier=self.speculators_config.verifier.name_or_path,
            target_layer_ids=self.aux_hidden_state_layer_ids,
            target_layer_start_idx=1,
            draft_vocab_size=self.draft_vocab_size,
            sample_from_anchor=self.sample_from_anchor,
            mask_token_id=self.mask_token_id,
            sliding_window_non_causal=self.sliding_window_non_causal,
            training_framework="vllm-project/speculators",
            training_framework_version=self.speculators_version,
        )
        metadata = {
            key: value
            for key, value in native.items()
            if key not in self.__class__.model_fields and key != "architectures"
        }
        metadata["verifier_architectures"] = (
            self.speculators_config.verifier.architectures
        )
        metadata["speculators_version"] = self.speculators_version
        return {
            **self.transformer_layer_config.to_dict(),
            "dtype": native.get("dtype"),
            "architectures": ["DflashDraftModel"],
            "open_spec_config_version": "0.0.0",
            "speculative_config": settings.model_dump(exclude_none=True),
            "target_hidden_size": self.target_hidden_size,
            "speculators_metadata": metadata,
        }

    def to_diff_dict(self) -> dict[str, Any]:
        """Keep the standalone wire config complete in config.json."""
        if (
            self.speculators_model_type == "dflash"
            and self.speculators_config is not None
        ):
            return self.to_dict()
        return super().to_diff_dict()
