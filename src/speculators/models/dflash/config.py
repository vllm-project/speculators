from typing import Any, Literal

import torch
from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from speculators import SpeculatorModelConfig

__all__ = [
    "DFlashSpeculatorConfig",
]

_MOE_TO_DENSE_MAP: dict[str, type[PretrainedConfig]] = {
    "qwen3_omni_moe_text": Qwen3Config,
}

_QWEN3_DENSE_WHITELIST = {
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "tie_word_embeddings",
    "torch_dtype",
    "use_qk_norm",
    "vocab_size",
}


@SpeculatorModelConfig.register("dflash")
class DFlashSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for DFlash speculator with vocabulary mapping.

    DFlash features vocabulary mapping between draft (64K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

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

    max_anchors: int = Field(
        default=256,
        description=(
            "Maximum number of anchor positions to sample during training "
            "(controls memory usage and training efficiency)"
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

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            model_type = value.get("model_type")
            if model_type in _MOE_TO_DENSE_MAP:
                config_class = _MOE_TO_DENSE_MAP[model_type]
                filtered_value = {
                    key: item
                    for key, item in value.items()
                    if key in _QWEN3_DENSE_WHITELIST
                }
                filtered_value["model_type"] = "qwen3"
                return config_class(**filtered_value)

            config_class: type[PretrainedConfig] = Qwen3Config
            if model_type is not None:
                config_class = AutoConfig.for_model(model_type=model_type).__class__
            return config_class(**value)
        return value

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size


DFlashSpeculatorConfig.model_rebuild(force=True, _types_namespace={"torch": torch})
