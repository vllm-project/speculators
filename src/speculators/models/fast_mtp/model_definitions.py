"""FastMTP layer and block definitions for multi-token prediction.

Architecture overview
---------------------
``MTPBlock`` is the top-level container for a single FastMTP prediction step.
Its module layout and state-dict keys mirror the original Qwen3-Next checkpoint
structure so that no key remapping is required between training, conversion, and
vLLM inference::

    mtp.pre_fc_norm_hidden.weight
    mtp.pre_fc_norm_embedding.weight
    mtp.fc.weight
    mtp.norm.weight
    mtp.layers.0.<all decoder weights>

``Qwen3NextFastMTPLayer`` is a thin subclass of ``Qwen3NextDecoderLayer`` that
selects the right layer index (a full-attention layer) and forces eager attention.
It is passed as ``layer_class`` to ``MTPBlock`` and lives under ``mtp.layers.0``.

Adding support for a new verifier model (e.g., Qwen3.5, DeepSeek) requires only:
1. A thin decoder-layer subclass that sets the correct attention implementation
   and layer index, analogous to ``Qwen3NextFastMTPLayer``.
2. An entry in ``fast_mtp_model_classes`` pointing to that subclass and its
   corresponding norm and rotary embedding classes.
``MTPBlock`` is model-agnostic and needs no changes.
"""

import copy

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
)

from speculators.models import base_components

__all__ = ["MTPBlock", "Qwen3NextFastMTPLayer", "fast_mtp_model_classes"]


class MTPBlock(nn.Module):
    """Single FastMTP prediction block.

    Fuses verifier hidden states with draft token embeddings, passes the
    result through one decoder layer, and applies a final layer norm.

    Module structure mirrors the Qwen3-Next ``mtp.*`` checkpoint layout so
    state-dict keys match vLLM directly — no remapping needed:

    * ``pre_fc_norm_hidden``   → ``mtp.pre_fc_norm_hidden.weight``
    * ``pre_fc_norm_embedding`` → ``mtp.pre_fc_norm_embedding.weight``
    * ``fc``                   → ``mtp.fc.weight``
    * ``norm``                 → ``mtp.norm.weight``
    * ``layers[0]``            → ``mtp.layers.0.<decoder_keys>``

    :param layer_class: Decoder-layer class to instantiate inside ``layers``.
    :param norm_class: RMSNorm class (accepts ``(hidden_size, eps=eps)``).
    :param config: Model config supplying ``hidden_size`` and ``rms_norm_eps``.
    """

    def __init__(
        self,
        layer_class: type,
        norm_class: type,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        hidden_size: int = config.hidden_size  # type: ignore[assignment]
        eps: float = config.rms_norm_eps  # type: ignore[assignment]
        self.pre_fc_norm_hidden = norm_class(hidden_size, eps=eps)
        self.pre_fc_norm_embedding = norm_class(hidden_size, eps=eps)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.norm = norm_class(hidden_size, eps=eps)
        self.layers = nn.ModuleList([layer_class(config)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the FastMTP block.

        :param hidden_states: Verifier hidden states ``[batch, valid_len, H]``.
        :param token_embeddings: Token embeddings ``[batch, valid_len, H]``.
        :param attention_mask: Optional attention mask.
        :param position_ids: Position IDs.
        :param position_embeddings: ``(cos, sin)`` tuple from rotary_emb.
        :returns: Output hidden states ``[batch, valid_len, H]``.
        """
        hidden_normed = self.pre_fc_norm_hidden(hidden_states)
        embed_normed = self.pre_fc_norm_embedding(token_embeddings)
        proj = self.fc(torch.cat([hidden_normed, embed_normed], dim=-1))
        output = self.layers[0](
            hidden_states=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden = output[0] if isinstance(output, tuple) else output
        return self.norm(hidden)


def _last_full_attention_idx(config: PretrainedConfig) -> int:
    """Return the last layer index whose type is ``full_attention``.

    Qwen3-Next alternates ``linear_attention`` (SSM/GatedDeltaNet) and
    ``full_attention`` layers.  The MTP head always uses standard self-attention,
    so the FastMTP layer must be instantiated with a ``full_attention`` index to
    get the correct weight structure.

    :param config: Model config with a ``layer_types`` list attribute.
    :returns: Index of the last ``full_attention`` layer in ``layer_types``.
    :raises ValueError: If no ``full_attention`` layer is found.
    """
    layer_types: list[str] = getattr(config, "layer_types", [])
    for i in reversed(range(len(layer_types))):
        if layer_types[i] == "full_attention":
            return i
    raise ValueError(
        "No full_attention layer found in config.layer_types. "
        f"layer_types={layer_types!r}. FastMTP requires a full_attention layer."
    )


class Qwen3NextFastMTPLayer(Qwen3NextDecoderLayer):  # type: ignore[misc]
    """Qwen3-Next decoder layer configured for the FastMTP head position.

    Forces eager attention (flash_attention_2 is not supported at the MTP
    position in Qwen3-Next's hybrid architecture) and selects the last
    full-attention layer index so the correct weight structure is initialised.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        # Shallow copy: only _attn_implementation (a string) is modified,
        # so no nested mutable objects are affected.
        modified = copy.copy(config)
        modified._attn_implementation = "eager"  # noqa: SLF001
        super().__init__(modified, _last_full_attention_idx(modified))  # type: ignore[arg-type]


fast_mtp_model_classes: dict[str, base_components.ModelComponents] = {
    "qwen3_next": base_components.override_components(
        "qwen3_next", first_layer_class=Qwen3NextFastMTPLayer
    ),
}
