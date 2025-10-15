from typing import NamedTuple, Optional
import torch
from transformers.configuration_utils import PretrainedConfig
import copy
from transformers import Cache, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs


class LlamaConcatInputDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: PretrainedConfig, layer_idx: int, norm_before_residual: bool = False):
        super().__init__(config, layer_idx)

        ##### CHANGES START #####
        self.norm_before_residual = norm_before_residual
        if layer_idx == 0:
            self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn.q_proj = torch.nn.Linear(
                2 * config.hidden_size,  # previous: config.hidden_size
                config.num_attention_heads * config.head_dim,
                bias=config.attention_bias,
            )
            self.self_attn.k_proj = torch.nn.Linear(
                2 * config.hidden_size,  # previous: config.hidden_size
                config.num_key_value_heads * config.head_dim,
                bias=config.attention_bias,
            )
            self.self_attn.v_proj = torch.nn.Linear(
                2 * config.hidden_size,  # previous: config.hidden_size
                config.num_key_value_heads * config.head_dim,
                bias=config.attention_bias,
            )
        self.layer_idx = layer_idx
        ##### CHANGES END #####

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        ##### CHANGES START #####
        # previous: residual = hidden_states
        if self.layer_idx == 0:
            # hidden_states are cat([embeds, hidden], dim=-1)
            # so residual should be hidden part only, and embeds should be normalized
            mid = hidden_states.shape[2] // 2
            embeds, hidden = hidden_states.split(mid, dim=-1)
            residual = hidden

            # Apply norms
            embeds = self.input_layernorm(embeds)
            hidden = self.hidden_norm(hidden)
            if self.norm_before_residual:
                residual = hidden # set residual to normalized hidden
            hidden_states = torch.cat([embeds, hidden], dim=-1)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        ##### CHANGES END #####

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaConcatInputRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: LlamaConfig, device=None):
        config = copy.copy(config)
        config.hidden_size = config.hidden_size * 2
        super().__init__(config, device)


class ModelComponents(NamedTuple):
    decoder_layer_class: type
    norm_class: type
    rotary_emb_class: type


model_classes: dict[str, ModelComponents] = {
    "llama": ModelComponents(
        LlamaConcatInputDecoderLayer, LlamaRMSNorm, LlamaConcatInputRotaryEmbedding
    ),
}
