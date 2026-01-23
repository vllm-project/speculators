"""Shared base model components for all speculator types."""

from typing import NamedTuple

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


class ModelComponents(NamedTuple):
    """Base components for a model architecture."""

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
    "qwen3": ModelComponents(
        Qwen3DecoderLayer,  # first_layer_class (same as decoder for base models)
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    ),
}


def override_components(model_type: str, **overrides) -> ModelComponents:
    """Create ModelComponents by overriding specific fields from base model_classes.

    Args:
        model_type: Base model type (e.g., "llama", "qwen3")
        **overrides: Fields to override (e.g., first_layer_class=MyCustomLayer)

    Returns:
        New ModelComponents with specified overrides
    """
    base = model_classes[model_type]
    return base._replace(**overrides)
