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

_MODEL_TYPE_ALIASES: dict[str, set[str]] = {}

_MIN_TRANSFORMERS_VERSION: dict[str, str] = {
    "qwen3_next": "4.57.0",
    "qwen3_5_text": "5.2.0",
    "qwen3_5_moe_text": "5.2.0",
}


def resolve_model_type(model_type: str) -> str:
    """Resolve a model_type to a canonical type in mtp_model_classes."""
    if model_type in mtp_model_classes:
        return model_type
    for canonical, aliases in _MODEL_TYPE_ALIASES.items():
        if model_type in aliases:
            return canonical
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

    Hybrid models (Qwen3-Next, Qwen3.5, Qwen3.6) alternate
    linear_attention and full_attention layers. The MTP head always
    uses full_attention, so we must pick a matching index.
    """
    layer_types: list[str] = getattr(config, "layer_types", [])
    for i in reversed(range(len(layer_types))):
        if layer_types[i] == "full_attention":
            return i
    return 0


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
