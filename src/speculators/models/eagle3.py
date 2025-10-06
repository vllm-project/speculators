"""
Speculators implementation of EAGLE-3:
    - https://arxiv.org/abs/2503.01840

Classes:
    Eagle3SpeculatorConfig: Configuration class for EAGLE-3 speculator model
    EagleSpeculator3: Main model implementation for EAGLE-3 speculators
    Eagle3Attention: Custom attention layer for EAGLE-3, processes
        concatenated embeddings and hidden states
    Eagle3DecoderLayer: Custom decoder layer for EAGLE-3, processes
        concatenated embeddings and hidden states with Eagle3Attention
        and support for moving hidden layernorm before residual
"""

import os
from typing import Any, ClassVar, Literal, Optional, Union

import torch
from pydantic import Field, field_serializer, field_validator
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from speculators import SpeculatorModel, SpeculatorModelConfig

__all__ = [
    "Eagle3Attention",
    "Eagle3DecoderLayer",
    "Eagle3Speculator",
    "Eagle3SpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle3")
class Eagle3SpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for EAGLE-3 speculator with vocabulary mapping.

    EAGLE-3 features vocabulary mapping between draft (32K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    :param norm_before_residual: Apply hidden_norm before storing residual
    """

    speculators_model_type: Literal["eagle3"] = "eagle3"
    architectures: list[str] = Field(
        default_factory=lambda: ["Eagle3Speculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    norm_before_residual: bool = Field(
        default=False,
        description="Apply hidden_norm before storing residual",
    )

    target_hidden_size: Optional[int] = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    eagle_aux_hidden_state_layer_ids: Optional[list[int]] = Field(
        default=None,
        description="Layer IDs of the Eagle auxiliary hidden state layers",
    )

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                from transformers import AutoConfig

                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value


class Eagle3Attention(nn.Module):
    """
    Eagle-3 attention module that processes concatenated embeddings and hidden states.

    Modified from standard Llama attention to accept 2x hidden_size input
    for Q/K/V projections while maintaining standard output size.
    """

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        input_size = 2 * self.hidden_size
        self.q_proj = nn.Linear(
            input_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            input_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            input_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple:
        """
        Forward pass for Eagle-3 attention.
        Taken from Llama Attention but modified to accept 2x hidden_size input.

        :param hidden_states: Input tensor of shape [batch, seq_len, 2*hidden_size]
        :param attention_mask: Optional attention mask
        :param position_ids: Optional position IDs for rotary embeddings
        :param past_key_value: Optional cached key-value pairs
        :param output_attentions: Whether to return attention weights
        :param use_cache: Whether to cache key-value pairs
        :param position_embeddings: Optional precomputed rotary embeddings
        :return: Tuple of (hidden_states, [attention_weights], [past_key_value])
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        past_key_value_out = None
        if past_key_value is not None:
            past_key = past_key_value[0]
            past_value = past_key_value[1]
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        if use_cache:
            past_key_value_out = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (
            self.head_dim**0.5
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value_out


class Eagle3DecoderLayer(nn.Module):
    """
    Eagle-3 decoder layer that processes concatenated embeddings and hidden states.

    Accepts 2x hidden_size input from concatenated embeddings and fused hidden states.
    Uses Eagle3Attention for the self-attention computation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm_before_residual = norm_before_residual

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.self_attn = Eagle3Attention(config, layer_idx)

        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,  # noqa: ARG002
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple:
        """
        Process concatenated embeddings and hidden states through modified decoder
        layer.

        :param hidden_states: Input tensor of shape [batch, seq_len, 2*hidden_size]
        :return: Tuple of layer outputs
        """
        embeds = hidden_states[:, :, : self.hidden_size]
        hidden = hidden_states[:, :, self.hidden_size : 2 * self.hidden_size]

        if self.norm_before_residual:
            hidden = self.hidden_norm(hidden)
            residual = hidden
        else:
            residual = hidden
            hidden = self.hidden_norm(hidden)

        embeds = self.input_layernorm(embeds)

        attn_input = torch.cat([embeds, hidden], dim=-1)

        attn_output, attn_weights, past_key_value_out = self.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # type: ignore[assignment]

        if use_cache:
            outputs += (past_key_value_out,)  # type: ignore[assignment]

        return outputs


@SpeculatorModel.register("eagle3")
class Eagle3Speculator(SpeculatorModel):
    """
    EAGLE-3 speculator with vocabulary mapping and multi-layer fusion.

    EAGLE-3 processes concatenated hidden states from multiple verifier layers
    through a fusion layer, then combines with embeddings for a custom decoder
    layer that accepts 2x hidden_size input.
    """

    config_class: ClassVar[type[Eagle3SpeculatorConfig]] = Eagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier*",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = []  # type: ignore[misc,assignment]

    def __init__(
        self,
        config: Eagle3SpeculatorConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full", "train_only"]
        ] = None,
        reduce_vocab_size: bool = True,
        has_drafter_embedding: bool = True,
    ):
        """
        Initialize Eagle3 speculator.

        :param config: Eagle3SpeculatorConfig instance
        :param verifier: Optional verifier model
        :param verifier_attachment_mode: How to attach the verifier
        :param reduce_vocab_size: Whether to reduce vocabulary size with mapping
        :param has_drafter_embedding: Whether drafter embedding weights are provided
        """
        if not isinstance(config, Eagle3SpeculatorConfig):
            raise ValueError(
                f"config must be Eagle3SpeculatorConfig, got {type(config)}"
            )

        self.config: Eagle3SpeculatorConfig = config

        self.hidden_size = config.transformer_layer_config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.target_vocab_size = config.target_vocab_size

        # Use target_hidden_size if specified, otherwise use draft model's hidden_size
        self.target_hidden_size = (
            config.target_hidden_size
            if config.target_hidden_size is not None
            else self.hidden_size
        )

        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
        )

        if has_drafter_embedding:
            self.embed_tokens = nn.Embedding(
                self.target_vocab_size,
                self.hidden_size,
                padding_idx=config.transformer_layer_config.pad_token_id
                if hasattr(config.transformer_layer_config, "pad_token_id")
                else None,
            )

        self.fc = nn.Linear(
            3 * self.target_hidden_size,  # Use target model's hidden size
            self.hidden_size,
            bias=False,
        )

        self.layers = nn.ModuleList(
            [
                Eagle3DecoderLayer(
                    config.transformer_layer_config,
                    layer_idx=0,
                    norm_before_residual=config.norm_before_residual,
                )
            ]
        )

        self.norm = LlamaRMSNorm(
            self.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            self.hidden_size,
            self.draft_vocab_size,
            bias=False,
        )
        if reduce_vocab_size:
            self.register_buffer(  # type: ignore[attr-defined]
                "d2t",
                torch.zeros(self.draft_vocab_size, dtype=torch.long),
            )
            self.register_buffer(  # type: ignore[attr-defined]
                "t2d",
                torch.zeros(self.target_vocab_size, dtype=torch.bool),
            )

            # Type hints for buffers
            self.d2t: torch.Tensor
            self.t2d: torch.Tensor
        self.post_init()  # type: ignore[attr-defined]

    def tie_weights(self):
        """
        Override tie_weights to prevent vocabulary corruption in transformers 4.54.1+

        Eagle3 intentionally uses different vocabulary sizes:
        - Input embeddings (embed_tokens): 128256 (full vocabulary)
        - Output embeddings (lm_head): 32000 (draft vocabulary)

        The default tie_weights() tries to make them identical, breaking Eagle3.
        This override preserves the intentional vocabulary size difference.
        """
        # Don't call super().tie_weights() - this prevents vocabulary corruption
        # that occurs when _tie_or_clone_weights replaces lm_head.weight with
        # embed_tokens.weight
