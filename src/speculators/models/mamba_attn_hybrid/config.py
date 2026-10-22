from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "MambaAttnHybridSpeculatorConfig",
]

BlockType = Literal["mamba", "attention", "mlp"]

DEFAULT_BLOCK_PATTERN: list[BlockType] = [
    "mamba",
    "mlp",
    "mamba",
    "attention",
    "mlp",
    "mamba",
    "attention",
    "mlp",
    "mamba",
    "attention",
    "mlp",
]


@SpeculatorModelConfig.register("mamba_attn_hybrid")
class MambaAttnHybridSpeculatorConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["mamba_attn_hybrid"] = "mamba_attn_hybrid"
    architectures: list[str] = Field(
        default_factory=lambda: ["MambaAttnHybridSpeculator"]
    )
    transformer_layer_config: PretrainedConfig = Field(default_factory=Qwen3Config)
    draft_vocab_size: int = 32000
    block_size: int = 8
    mask_token_id: int | None = None
    sliding_window_non_causal: bool = False
    block_pattern: list[BlockType] = Field(
        default_factory=lambda: list(DEFAULT_BLOCK_PATTERN)
    )
    attn_kv_layer_ids: list[int] = Field(default_factory=list)
    latent_fusion_layer_ids: list[int] = Field(default_factory=list)
    mamba_num_heads: int = 48
    mamba_head_dim: int = 64
    mamba_d_state: int = 4
    mamba_n_groups: int = 4
    mamba_conv_kernel: int = 4
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            return AutoConfig.for_model(**{"model_type": "qwen3", **value})
        return value
