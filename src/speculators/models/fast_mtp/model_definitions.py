"""FastMTP layer mixin and Qwen3-Next concrete layer class.

FastMTP projects hidden states DOWN to standard hidden_size BEFORE attention via
input_proj, so q/k/v weights remain standard width. The mixin interface takes two
separate tensors (hidden_states, token_embeddings).
"""

import copy

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
)

from speculators.models import base_components

__all__ = ["FastMTPLayerMixin", "Qwen3NextFastMTPLayer", "fast_mtp_model_classes"]


class FastMTPLayerMixin:
    """FastMTP-specific modifications for any decoder layer.

    Projects hidden states to standard hidden_size BEFORE attention via input_proj;
    q/k/v weights remain standard width.
    """

    # Declared for type checkers; provided by the base decoder layer class
    self_attn: nn.Module
    mlp: nn.Module
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module

    def _setup_fastmtp_modules(
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through FastMTP layer.

        :param hidden_states: Verifier hidden states [batch, valid_len, hidden_size]
        :param token_embeddings: Token embeddings [batch, valid_len, hidden_size]
        :param attention_mask: Optional attention mask
        :param position_ids: Position IDs
        :param position_embeddings: (cos, sin) tuple from rotary_emb; required because
            Qwen3NextAttention destructures it directly.
        :param kwargs: Additional arguments forwarded to the base decoder layer
        :return: Output hidden states [batch, valid_len, hidden_size]
        """
        hidden_normed = self.hidden_layernorm(hidden_states)
        embed_normed = self.token_layernorm(token_embeddings)
        proj = self.input_proj(torch.cat([hidden_normed, embed_normed], dim=-1))

        output = super().forward(  # type: ignore[misc]
            hidden_states=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        return self.final_layernorm(hidden_states)


def _last_full_attention_idx(config: PretrainedConfig) -> int:
    """Return the last layer index whose type is ``full_attention``.

    Qwen3-Next alternates linear_attention and full_attention layers. The MTP
    head in the checkpoint always uses full_attention, so we must instantiate the
    decoder layer with a full_attention index — otherwise Qwen3NextDecoderLayer
    creates a linear_attn (SSM/GatedDeltaNet) layer instead.
    """
    layer_types: list[str] = getattr(config, "layer_types", [])
    for i in reversed(range(len(layer_types))):
        if layer_types[i] == "full_attention":
            return i
    return 0


class Qwen3NextFastMTPLayer(FastMTPLayerMixin, Qwen3NextDecoderLayer):  # type: ignore[misc]
    """FastMTP layer for Qwen3-Next (sparse MoE) checkpoints."""

    def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:  # noqa: ARG002
        modified = copy.copy(config)
        modified._attn_implementation = "eager"  # noqa: SLF001
        super().__init__(modified, _last_full_attention_idx(modified))  # type: ignore[arg-type]
        self._setup_fastmtp_modules(modified, Qwen3NextRMSNorm)


fast_mtp_model_classes: dict[str, base_components.ModelComponents] = {
    "qwen3_next": base_components.override_components(
        "qwen3_next", first_layer_class=Qwen3NextFastMTPLayer
    ),
}
