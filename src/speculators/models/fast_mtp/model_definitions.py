"""FastMTP layer mixin and Qwen3-Next concrete layer class.

The FastMTP layer fuses verifier hidden states with embedded draft tokens:

1. Normalize each input independently
   (``pre_fc_norm_hidden``, ``pre_fc_norm_embedding``)
2. Concatenate the two normalized tensors along the hidden dimension
3. Project 2H → H via a single linear layer (``fc``)
4. Pass through a standard Qwen3-Next decoder block (full self-attention + MLP)
5. Apply a final layer norm (``norm``)

Attribute names (``pre_fc_norm_hidden``, ``pre_fc_norm_embedding``, ``fc``, ``norm``)
are chosen to match Qwen3-Next's original MTP weight keys exactly.  This makes the
speculators ↔ Qwen3-Next key remapping in
:mod:`~speculators.models.fast_mtp.checkpoint` a trivial frozenset lookup with no
explicit mapping dict.
"""

import copy
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
)

from speculators.models import base_components

__all__ = ["FastMTPLayerMixin", "Qwen3NextFastMTPLayer", "fast_mtp_model_classes"]

# The projection input is the concatenation of the hidden state and the token
# embedding, each of size hidden_size, so the input dimension is 2 * hidden_size.
_CONCAT_PROJECTION_FACTOR = 2


@runtime_checkable
class _DecoderLayerInterface(Protocol):
    """Interface that :class:`FastMTPLayerMixin` requires from its base decoder class.

    Type checkers can verify that any class mixed with
    :class:`FastMTPLayerMixin` exposes these standard decoder-layer attributes.
    """

    self_attn: nn.Module
    mlp: nn.Module
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module


class FastMTPLayerMixin:
    """FastMTP-specific modifications for any decoder layer.

    Normalizes verifier hidden states and token embeddings separately, then
    concatenates and projects them to the standard hidden dimension before
    feeding into the underlying decoder block.  All standard decoder-layer
    attributes (``self_attn``, ``mlp``, ``input_layernorm``,
    ``post_attention_layernorm``) must be provided by the base class.

    Attribute names match Qwen3-Next's original MTP weight keys so that the
    speculators ↔ Qwen3-Next key remapping requires no explicit dict.
    """

    # Required by the base decoder layer class; declared here for type checkers.
    self_attn: nn.Module
    mlp: nn.Module
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module

    def _setup_fastmtp_modules(
        self,
        config: PretrainedConfig,
        norm_class: type[nn.Module],
    ) -> None:
        """Initialize the four FastMTP-specific modules.

        Must be called from ``__init__`` after the base decoder layer is
        constructed.  ``norm_class`` must accept ``(hidden_size, eps=eps)``
        arguments (e.g. ``Qwen3NextRMSNorm``).

        :param config: Model config providing ``hidden_size`` and
            ``rms_norm_eps``.
        :param norm_class: RMSNorm class to use for the three layer norms.
        """
        hidden_size: int = config.hidden_size  # type: ignore[assignment]
        eps: float = config.rms_norm_eps  # type: ignore[assignment]
        # Names match Qwen3-Next original mtp.* keys; see checkpoint.py.
        self.pre_fc_norm_hidden = norm_class(hidden_size, eps=eps)
        self.pre_fc_norm_embedding = norm_class(hidden_size, eps=eps)
        self.fc = nn.Linear(
            _CONCAT_PROJECTION_FACTOR * hidden_size, hidden_size, bias=False
        )
        self.norm = norm_class(hidden_size, eps=eps)

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
        hidden_normed = self.pre_fc_norm_hidden(hidden_states)
        embed_normed = self.pre_fc_norm_embedding(token_embeddings)
        proj = self.fc(torch.cat([hidden_normed, embed_normed], dim=-1))

        output = super().forward(  # type: ignore[misc]
            hidden_states=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        return self.norm(hidden_states)


def _last_full_attention_idx(config: PretrainedConfig) -> int:
    """Return the last layer index whose type is ``full_attention``.

    Qwen3-Next alternates ``linear_attention`` (SSM/GatedDeltaNet) and
    ``full_attention`` layers at a fixed interval controlled by
    ``full_attention_interval``.  The MTP head in the Qwen3-Next checkpoint
    always uses standard self-attention (full_attention), so the FastMTP
    layer must be instantiated with a ``full_attention`` layer index.
    Using a ``linear_attention`` index would create an SSM layer instead of
    the required self-attention layer, silently producing wrong results.

    :param config: Model config with a ``layer_types`` list attribute.
    :returns: Index of the last ``full_attention`` layer in ``layer_types``.
    :raises ValueError: If ``layer_types`` contains no ``full_attention``
        entry (unexpected architecture).
    """
    layer_types: list[str] = getattr(config, "layer_types", [])
    for i in reversed(range(len(layer_types))):
        if layer_types[i] == "full_attention":
            return i
    raise ValueError(
        "No full_attention layer found in config.layer_types. "
        f"layer_types={layer_types!r}. FastMTP requires a full_attention layer "
        "to instantiate standard self-attention weights."
    )


class Qwen3NextFastMTPLayer(FastMTPLayerMixin, Qwen3NextDecoderLayer):  # type: ignore[misc]
    """FastMTP layer for Qwen3-Next (sparse MoE) checkpoints."""

    def __init__(self, config: PretrainedConfig, layer_idx: int = 0) -> None:  # noqa: ARG002
        # Shallow copy suffices: only _attn_implementation (a string scalar) is
        # modified on the copy, so no nested mutable objects are affected.
        modified = copy.copy(config)
        # Force eager attention: Qwen3-Next's hybrid decoder layer does not
        # support flash_attention_2 at the MTP position.
        modified._attn_implementation = "eager"  # noqa: SLF001
        super().__init__(modified, _last_full_attention_idx(modified))  # type: ignore[arg-type]
        self._setup_fastmtp_modules(modified, Qwen3NextRMSNorm)


fast_mtp_model_classes: dict[str, base_components.ModelComponents] = {
    "qwen3_next": base_components.override_components(
        "qwen3_next", first_layer_class=Qwen3NextFastMTPLayer
    ),
}
