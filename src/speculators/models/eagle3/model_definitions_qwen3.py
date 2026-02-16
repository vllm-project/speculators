import torch
from transformers import Cache, PretrainedConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from speculators.models.eagle3.model_components import ModelComponents


def _ensure_qwen3_llama_compat(config: PretrainedConfig) -> None:
    # Qwen3-VL text config may omit fields read directly by HF Llama layers.
    if not hasattr(config, "mlp_bias"):
        setattr(config, "mlp_bias", False)
    if not hasattr(config, "attention_bias"):
        setattr(config, "attention_bias", False)


class Qwen3DecoderEagle3FirstLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        _ensure_qwen3_llama_compat(config)
        super().__init__(config, layer_idx)

        self.norm_before_residual = norm_before_residual
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn.q_proj = torch.nn.Linear(
            2 * config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.k_proj = torch.nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.v_proj = torch.nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore[valid-type]
    ) -> torch.Tensor:
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        _ensure_qwen3_llama_compat(config)
        super().__init__(config, layer_idx)


QWEN3_MODEL_COMPONENTS = ModelComponents(
    first_layer_class=Qwen3DecoderEagle3FirstLayer,
    decoder_layer_class=Qwen3DecoderLayer,
    norm_class=LlamaRMSNorm,
    rotary_emb_class=LlamaRotaryEmbedding,
)

