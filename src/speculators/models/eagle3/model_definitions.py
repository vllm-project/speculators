from typing import Any

import torch
from transformers import Cache, LlamaConfig, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from speculators.models import base_components


class Eagle3TTTCacheMixin:
    """Keep the TTT KV cache alive under gradient checkpointing.

    Eagle3 unrolls ``ttt_steps`` passes over the same sequence and step ``s``
    attends over the K/V of steps ``0..s`` through a shared ``DynamicCache``.
    transformers' ``GradientCheckpointingLayer`` nulls ``past_key_values`` for
    any layer that does not opt in via ``_can_checkpoint_with_cache``, which
    leaves the layer with ``kv_len == q_len`` while the block mask was built
    for the grown cache. Every decoder layer in the draft needs this, not just
    the first one.

    The backward replay of a checkpointed layer runs after later TTT steps
    have appended to the cache, so before the replayed forward appends again
    this layer's entry is cropped back to ``cache_position[0]``. Only this
    layer's entry is touched: autograd gives no ordering guarantee across
    layers, and cropping the whole cache from one layer could discard entries
    another layer has not replayed yet.
    """

    _can_checkpoint_with_cache = True

    # Provided by the decoder layer base class
    self_attn: Any
    gradient_checkpointing: bool
    training: bool

    def _crop_ttt_cache(
        self, past_key_values: Cache | None, cache_position: torch.LongTensor | None
    ) -> None:
        # Same gate transformers uses to decide a layer is being checkpointed.
        # Outside it the cache only ever grows in step with cache_position, and
        # skipping the check avoids a GPU sync / dynamo graph break per layer in
        # the compiled (non-checkpointed) forward.
        if not (self.gradient_checkpointing and self.training):
            return
        if past_key_values is None or cache_position is None:
            return
        layer_idx = self.self_attn.layer_idx
        layers = getattr(past_key_values, "layers", None)
        if layers is None or layer_idx >= len(layers):
            return
        extra = past_key_values.get_seq_length(layer_idx) - int(cache_position[0])
        if extra > 0:
            layers[layer_idx].crop(-extra)


class Eagle3FirstLayerMixin(Eagle3TTTCacheMixin):
    """Shared Eagle3 first-layer modifications for any decoder layer.

    Patches q/k/v projections to accept 2x hidden_size input (cat([embeds, hidden]))
    and overrides forward to split, normalize, and recombine before attention.
    """

    # Provided by the decoder layer base class
    input_layernorm: Any
    post_attention_layernorm: Any
    mlp: Any
    norm_before_residual: bool
    hidden_norm: Any

    def _patch_eagle3_projections(
        self,
        config: PretrainedConfig,
        norm_class: type[torch.nn.Module],
        norm_before_residual: bool,
    ):
        """Replace q/k/v projections with 2x hidden_size input and add hidden_norm."""
        self.norm_before_residual = norm_before_residual
        self.hidden_norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
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
        # Previously in the parent DecoderLayer:
        #   residual = hidden_states
        #   hidden_states = self.input_layernorm(hidden_states)

        # ##### Start of Eagle3 modifications #####
        self._crop_ttt_cache(past_key_values, cache_position)
        # hidden_states are cat([embeds, hidden], dim=-1)
        # so residual should be hidden part only, and embeds should be normalized
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        # Apply norms
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden  # set residual to normalized hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)
        if torch.__version__ >= "2.10":
            # As of `torch==2.10`, compile attempts to fuse together too many
            # ops, resulting in a fused kernel that exceeds shared memory limits
            # For now, we force a graph break to prevent this
            # https://github.com/pytorch/pytorch/issues/175250
            torch._dynamo.graph_break()  # noqa: SLF001

        # ##### End of Eagle3 modifications #####

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
        return hidden_states  # noqa: RET504


class Eagle3DecoderLayerMixin(Eagle3TTTCacheMixin):
    """Stock decoder layer (layers 1..N-1) keeping the TTT cache when checkpointed."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._crop_ttt_cache(past_key_values, cache_position)
        return super().forward(  # type: ignore[misc]
            hidden_states,
            *args,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )


class LlamaDecoderEagle3Layer(Eagle3DecoderLayerMixin, LlamaDecoderLayer):  # type:ignore[misc]
    pass


class Qwen3DecoderEagle3Layer(Eagle3DecoderLayerMixin, Qwen3DecoderLayer):  # type:ignore[misc]
    pass


class LlamaDecoderEagle3FirstLayer(Eagle3FirstLayerMixin, LlamaDecoderLayer):  # type:ignore[misc]
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, LlamaRMSNorm, norm_before_residual)


class Qwen3DecoderEagle3FirstLayer(Eagle3FirstLayerMixin, Qwen3DecoderLayer):  # type:ignore[misc]
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, Qwen3RMSNorm, norm_before_residual)


model_classes: dict[str, base_components.ModelComponents] = {
    "llama": base_components.override_components(
        "llama",
        first_layer_class=LlamaDecoderEagle3FirstLayer,
        decoder_layer_class=LlamaDecoderEagle3Layer,
    ),
    "qwen3": base_components.override_components(
        "qwen3",
        first_layer_class=Qwen3DecoderEagle3FirstLayer,
        decoder_layer_class=Qwen3DecoderEagle3Layer,
    ),
}
