"""Shared base model components for all speculator types."""

from typing import NamedTuple

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
        Qwen3NextRotaryEmbedding,
    )

    HAS_QWEN3_NEXT = True
except ImportError:
    HAS_QWEN3_NEXT = False


class ModelComponents(NamedTuple):
    """Container for the components of a speculators model.

    This groups the building blocks needed to construct a model, enabling
    architecture-agnostic code and selective component overriding for
    speculative decoding algorithms.

    Attributes:
        first_layer_class: Class for the first decoder layer. Can be customized
            for speculative decoding while keeping other layers standard.
        decoder_layer_class: Class for standard decoder layers used throughout
            the rest of the model.
        norm_class: Normalization layer class (e.g., LlamaRMSNorm, Qwen3RMSNorm).
        rotary_emb_class: Rotary positional embedding class for the model.
    """

    first_layer_class: type
    decoder_layer_class: type
    norm_class: type
    rotary_emb_class: type


model_classes: dict[str, ModelComponents] = {
    "llama": ModelComponents(
        LlamaDecoderLayer,  # first_layer_class (same as decoder for base models)
        LlamaDecoderLayer,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
    ),
    "qwen2": ModelComponents(
        Qwen2DecoderLayer,
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
        Qwen2RotaryEmbedding,
    ),
    "qwen3": ModelComponents(
        Qwen3DecoderLayer,  # first_layer_class (same as decoder for base models)
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    ),
}

if HAS_QWEN3_NEXT:
    model_classes["qwen3_next"] = ModelComponents(
        Qwen3NextDecoderLayer,
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
        Qwen3NextRotaryEmbedding,
    )


def override_components(model_type: str, **overrides) -> ModelComponents:
    """Override specific components from a base model architecture.

    Used for speculative decoding to swap custom layers (typically first_layer_class)
    while inheriting other components from the base model.

    Args:
        model_type: Base model type (e.g. "llama", "qwen2", "qwen3", "qwen3_next").
        **overrides: Component fields to override (first_layer_class,
            decoder_layer_class, etc).

    Returns:
        ModelComponents with specified overrides applied.
    """
    base = model_classes[model_type]
    return base._replace(**overrides)
