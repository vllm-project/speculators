from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from speculators import SpeculatorModelConfig

__all__ = [
    "MambaAttnHybridBaseConfig",
]


class MambaAttnHybridBaseConfig(SpeculatorModelConfig):
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

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Verifier layer IDs to extract KV caches from (one per draft layer)",
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

    sliding_window_prefix: int = Field(
        default=0,
        description="Number of attention-sink tokens at the start of each document "
        "to always attend to in sliding window layers (0 = disabled).",
    )

    trainable_q_proj: bool = Field(
        default=False,
        description="If True, draft q_proj is trainable. If False (default), frozen.",
    )
    trainable_kv_proj: bool = Field(
        default=False,
        description="If True, draft k_proj/v_proj are trainable. If False (default), frozen.",
    )
    trainable_q_norm: bool = Field(
        default=False,
        description="If True, draft q_norm is trainable. If False (default), frozen.",
    )
    trainable_k_norm: bool = Field(
        default=False,
        description="If True, draft k_norm is trainable. If False (default), frozen.",
    )

    init_q_proj: Literal["verifier", "random"] = Field(
        default="verifier",
        description="How to initialize q_proj: 'verifier' copies from target verifier "
        "layer, 'random' keeps PyTorch default init.",
    )
    init_kv_proj: Literal["verifier", "random"] = Field(
        default="verifier",
        description="How to initialize k_proj/v_proj: 'verifier' copies from target "
        "verifier layer, 'random' keeps PyTorch default init.",
    )
    init_q_norm: Literal["verifier", "random"] = Field(
        default="verifier",
        description="How to initialize q_norm: 'verifier' copies from target verifier "
        "layer, 'random' keeps PyTorch default init.",
    )
    init_k_norm: Literal["verifier", "random"] = Field(
        default="verifier",
        description="How to initialize k_norm: 'verifier' copies from target verifier "
        "layer, 'random' keeps PyTorch default init.",
    )

    layer_mapping: list[list[int]] | None = Field(
        default=None,
        description="Per-draft-layer verifier layer mapping. Each inner list is a group "
        "of verifier layer IDs: single-element groups use KV as-is, multi-element "
        "groups are aggregated according to agg_mode.",
    )

    agg_mode: Literal["avg", "learnedAvg", "headmix"] = Field(
        default="avg",
        description="Aggregation mode for multi-element layer_mapping groups. "
        "'avg' = uniform averaging, 'learnedAvg' = learned softmax weights, "
        "'headmix' = take disjoint KV head subsets from each verifier layer.",
    )

    no_mlp_layer_indices: list[int] | None = Field(
        default=None,
        description="Draft layer indices that skip the MLP block entirely "
        "(attention-only layers). None means all layers have MLP.",
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
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
        return self.transformer_layer_config.vocab_size
