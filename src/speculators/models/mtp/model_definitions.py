"""MTP layer mixin and concrete layer classes.

MTP projects hidden states DOWN to standard hidden_size BEFORE attention via
input_proj, so q/k/v weights remain standard width. The mixin interface takes
two separate tensors (hidden_states, token_embeddings).
"""

import copy
from importlib.metadata import version as pkg_version
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
)

from speculators.models import base_components

__all__ = ["MTPLayerMixin", "mtp_model_classes", "resolve_model_type"]

_MIN_TRANSFORMERS_VERSION: dict[str, str] = {
    "qwen3_next": "4.57.0",
    "qwen3_5_text": "5.2.0",
    "qwen3_5_moe_text": "5.2.0",
}


def resolve_model_type(model_type: str) -> str:
    """Resolve a model_type to a canonical type in mtp_model_classes."""
    if model_type in mtp_model_classes:
        return model_type
    raise _unsupported_model_type_error(model_type)


def _unsupported_model_type_error(model_type: str) -> ValueError:
    """Build a descriptive error for an unresolvable model type."""
    if model_type in _MIN_TRANSFORMERS_VERSION:
        min_ver = _MIN_TRANSFORMERS_VERSION[model_type]
        installed = pkg_version("transformers")
        return ValueError(
            f"Model type '{model_type}' requires transformers>={min_ver} "
            f"(installed: {installed}). "
            f"Upgrade with: pip install 'transformers>={min_ver}'"
        )
    supported = sorted(mtp_model_classes.keys())
    return ValueError(
        f"Unsupported MTP model type '{model_type}'. Supported types: {supported}"
    )


def _last_full_attention_idx(config: PretrainedConfig) -> int:
    """Return the last layer index whose type is ``full_attention``.

    Hybrid models (Qwen3-Next, Qwen3.5-MoE) alternate
    linear_attention and full_attention layers. The MTP head always
    uses full_attention, so we must pick a matching index.
    """
    layer_types: list[str] = getattr(config, "layer_types", [])
    if not layer_types:
        return 0
    for i in reversed(range(len(layer_types))):
        if layer_types[i] == "full_attention":
            return i
    raise ValueError(
        "Hybrid MTP layer requires at least one 'full_attention' entry in "
        f"config.layer_types, got: {layer_types}"
    )


class MTPLayerMixin:
    """MTP-specific modifications for any decoder layer.

    Projects hidden states to standard hidden_size BEFORE attention
    via input_proj; q/k/v weights remain standard width.
    """

    # Provided by the decoder layer base class
    self_attn: Any
    mlp: Any
    input_layernorm: Any
    post_attention_layernorm: Any

    def _setup_mtp_modules(
        self, config: PretrainedConfig, norm_class: type[nn.Module]
    ) -> None:
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.hidden_layernorm = norm_class(hidden_size, eps=eps)
        self.token_layernorm = norm_class(hidden_size, eps=eps)
        self.input_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.final_layernorm = norm_class(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: (tuple[torch.Tensor, torch.Tensor] | None) = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_normed = self.hidden_layernorm(hidden_states)
        embed_normed = self.token_layernorm(token_embeddings)
        proj = self.input_proj(torch.cat([embed_normed, hidden_normed], dim=-1))

        output = super().forward(  # type: ignore[misc]
            hidden_states=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        return self.final_layernorm(hidden_states)


def _create_mtp_layer_class(
    name: str,
    decoder_class: type,
    norm_class: type[nn.Module],
    *,
    hybrid: bool = False,
) -> type:
    """Create an MTP layer class for a specific decoder architecture.

    :param name: Class name for the generated type (used in repr/debugging).
    :param decoder_class: Base decoder layer class to mix with MTPLayerMixin.
    :param norm_class: RMSNorm class for the architecture.
    :param hybrid: If True, select the last full_attention layer index
        (for models that alternate linear/full attention layers).
    """

    class _MTPLayer(MTPLayerMixin, decoder_class):  # type: ignore[misc]
        def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
            modified = copy.copy(config)
            if hybrid:
                layer_idx = _last_full_attention_idx(modified)
            super().__init__(modified, layer_idx)  # type: ignore[arg-type]
            self._setup_mtp_modules(modified, norm_class)

    _MTPLayer.__name__ = name
    _MTPLayer.__qualname__ = name
    return _MTPLayer


Qwen3MTPLayer = _create_mtp_layer_class(
    "Qwen3MTPLayer", Qwen3DecoderLayer, Qwen3RMSNorm
)

mtp_model_classes: dict[str, base_components.ModelComponents] = {
    "qwen3": base_components.override_components(
        "qwen3", first_layer_class=Qwen3MTPLayer
    ),
}


if "qwen3_next" in base_components.model_classes:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
    )

    Qwen3NextMTPLayer = _create_mtp_layer_class(
        "Qwen3NextMTPLayer",
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
        hybrid=True,
    )
    mtp_model_classes["qwen3_next"] = base_components.override_components(
        "qwen3_next", first_layer_class=Qwen3NextMTPLayer
    )

if "qwen3_5_text" in base_components.model_classes:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
    )

    Qwen35MTPLayer = _create_mtp_layer_class(
        "Qwen35MTPLayer",
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        hybrid=True,
    )
    mtp_model_classes["qwen3_5_text"] = base_components.override_components(
        "qwen3_5_text", first_layer_class=Qwen35MTPLayer
    )

if "qwen3_5_moe_text" in base_components.model_classes:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
    )

    Qwen35MoeMTPLayer = _create_mtp_layer_class(
        "Qwen35MoeMTPLayer",
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
        hybrid=True,
    )
    mtp_model_classes["qwen3_5_moe_text"] = base_components.override_components(
        "qwen3_5_moe_text", first_layer_class=Qwen35MoeMTPLayer
    )

if "gemma2" in base_components.model_classes:
    from typing import Optional, Tuple, Any
    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2DecoderLayer,
        Gemma2RMSNorm,
        Gemma2Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS
    )

    class QueryOnlyGemma2Attention(Gemma2Attention):
        def __init__(self, config, layer_idx: Optional[int] = None):
            super().__init__(config, layer_idx)
            # Strip out K/V projections since we use verifier's KV cache
            self.k_proj = None
            self.v_proj = None

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            verifier_kv_last_local: Optional[torch.Tensor] = None,
            verifier_kv_last_global: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            is_local = (self.sliding_window is not None and self.sliding_window > 0)
            kv_tensor = verifier_kv_last_local if is_local else verifier_kv_last_global

            if kv_tensor is None or kv_tensor.dim() < 3:
                batch_sz, seq_len = input_shape
                kv_tensor = torch.zeros(
                    (batch_sz, seq_len, 2, self.num_key_value_heads, self.head_dim),
                    dtype=query_states.dtype,
                    device=query_states.device
                )
            elif kv_tensor.dim() == 3:
                batch_sz, seq_len, _ = kv_tensor.shape
                kv_tensor = kv_tensor.view(batch_sz, seq_len, 2, self.num_key_value_heads, self.head_dim)

            key_states = kv_tensor[..., 0, :, :].transpose(1, 2)
            value_states = kv_tensor[..., 1, :, :].transpose(1, 2)

            cos, sin = position_embeddings
            # Only apply RoPE to queries; verifier's KV cache already has RoPE applied
            query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            attention_interface = eager_attention_forward
            if getattr(self.config, "_attn_implementation", "eager") != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            # In eager mode, sliding_window is often dropped by the interface.
            # We enforce it directly onto the causal mask here if configured.
            if self.sliding_window is not None and self.sliding_window > 0 and attention_mask is not None:
                seq_len = attention_mask.shape[-1]
                window_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device))
                window_mask = torch.triu(window_mask, diagonal=-self.sliding_window + 1)
                attention_mask = attention_mask.masked_fill(~window_mask.view(1, 1, seq_len, seq_len), torch.finfo(query_states.dtype).min)

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                softcap=self.attn_logit_softcapping,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    class Gemma2MTPLayer(MTPLayerMixin, Gemma2DecoderLayer):
        def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
            super().__init__(config, layer_idx)
            self._setup_mtp_modules(config, Gemma2RMSNorm)
            # Replace the standard attention with query-only attention
            self.self_attn = QueryOnlyGemma2Attention(config, layer_idx)

    mtp_model_classes["gemma2"] = base_components.override_components(
        "gemma2", first_layer_class=Gemma2MTPLayer
    )

if "gemma4_text" in base_components.model_classes:
    from typing import Optional, Tuple, Any
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer,
        Gemma4RMSNorm,
        Gemma4TextAttention,
        Gemma4TextRotaryEmbedding,
        apply_rotary_pos_emb as gemma4_apply_rotary_pos_emb,
        eager_attention_forward as gemma4_eager_attention_forward,
    )

    class Gemma4MTPRotaryEmbedding(Gemma4TextRotaryEmbedding):
        """Wraps Gemma4TextRotaryEmbedding with a fixed layer_type for MTP.

        Gemma4TextRotaryEmbedding.forward requires a layer_type arg
        (per-type RoPE params), but MTPDraftModel.forward calls
        rotary_emb(x, position_ids) without one. This wrapper fixes
        the layer_type to full_attention — the single MTP layer must
        be full_attention (enforced by Gemma4TextConfig.__post_init__).
        """

        def __init__(self, config: PretrainedConfig, device=None) -> None:
            super().__init__(config, device)
            self._mtp_layer_type = "full_attention"

        def forward(
            self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return super().forward(x, position_ids, layer_type=self._mtp_layer_type)

    class QueryOnlyGemma4TextAttention(Gemma4TextAttention):
        def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
            super().__init__(config, layer_idx)
            self.k_proj = None
            self.v_proj = None
            if hasattr(self, "k_norm"):
                self.k_norm = None
            if hasattr(self, "v_norm"):
                self.v_norm = None
            self._num_kv_heads = config.num_attention_heads // self.num_key_value_groups

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            verifier_kv_last_local: Optional[torch.Tensor] = None,
            verifier_kv_last_global: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape)
            query_states = self.q_norm(query_states)

            cos, sin = position_embeddings
            query_states = gemma4_apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
            query_states = query_states.transpose(1, 2)

            kv_tensor = verifier_kv_last_local if self.is_sliding else verifier_kv_last_global

            if kv_tensor is None or kv_tensor.dim() < 3:
                batch_sz, seq_len = input_shape
                kv_tensor = torch.zeros(
                    (batch_sz, seq_len, 2, self._num_kv_heads, self.head_dim),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
            elif kv_tensor.dim() == 3:
                batch_sz, seq_len, _ = kv_tensor.shape
                kv_tensor = kv_tensor.view(
                    batch_sz, seq_len, 2, self._num_kv_heads, self.head_dim
                )

            key_states = kv_tensor[..., 0, :, :].transpose(1, 2)
            value_states = kv_tensor[..., 1, :, :].transpose(1, 2)

            if self.is_sliding and attention_mask is not None:
                seq_len = attention_mask.shape[-1]
                window_mask = torch.tril(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device)
                )
                window_mask = torch.triu(window_mask, diagonal=-self.sliding_window + 1)
                attention_mask = attention_mask.masked_fill(
                    ~window_mask.view(1, 1, seq_len, seq_len),
                    torch.finfo(query_states.dtype).min,
                )

            attn_output, attn_weights = gemma4_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    class Gemma4NativeMTPLayer(Gemma4TextDecoderLayer):
        """Gemma4 MTP layer matching vLLM's native Gemma4MultiTokenPredictor.

        Uses pre/post projection (no pre-fc norms) and returns a tuple
        (draft_hidden, backbone_hidden) for split logit/feedback paths.
        The single MTP layer is always full_attention — Gemma4TextConfig
        enforces this in __post_init__, and vLLM builds the model
        accordingly.
        """

        def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:
            config = copy.copy(config)
            config.layer_types = ["full_attention"]
            super().__init__(config, layer_idx)
            self.self_attn = QueryOnlyGemma4TextAttention(config, layer_idx)

            hidden_size = config.hidden_size
            self.pre_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.post_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.final_norm = Gemma4RMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.register_buffer(
                "normalizer",
                torch.tensor(hidden_size**0.5),
                persistent=False,
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            token_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            scaled_embeds = token_embeddings * self.normalizer
            combined = torch.cat([scaled_embeds, hidden_states], dim=-1)
            projected = self.pre_projection(combined)

            decoder_output = super().forward(
                hidden_states=projected,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            hidden = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output

            draft_hidden = self.final_norm(hidden)
            backbone_hidden = self.post_projection(draft_hidden)
            return draft_hidden, backbone_hidden

    mtp_model_classes["gemma4_text"] = base_components.override_components(
        "gemma4_text",
        first_layer_class=Gemma4NativeMTPLayer,
        rotary_emb_class=Gemma4MTPRotaryEmbedding,
    )
