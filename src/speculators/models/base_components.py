"""Shared base model components for all speculator types."""

from contextlib import suppress
from typing import NamedTuple

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


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
        first_layer_class=LlamaDecoderLayer,
        decoder_layer_class=LlamaDecoderLayer,
        norm_class=LlamaRMSNorm,
        rotary_emb_class=LlamaRotaryEmbedding,
    ),
    "qwen3": ModelComponents(
        first_layer_class=Qwen3DecoderLayer,
        decoder_layer_class=Qwen3DecoderLayer,
        norm_class=Qwen3RMSNorm,
        rotary_emb_class=Qwen3RotaryEmbedding,
    ),
}


with suppress(ImportError):
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextRMSNorm,
        Qwen3NextRotaryEmbedding,
    )

    model_classes["qwen3_next"] = ModelComponents(
        first_layer_class=Qwen3NextDecoderLayer,
        decoder_layer_class=Qwen3NextDecoderLayer,
        norm_class=Qwen3NextRMSNorm,
        rotary_emb_class=Qwen3NextRotaryEmbedding,
    )

with suppress(ImportError):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5TextRotaryEmbedding,
    )

    model_classes["qwen3_5_text"] = ModelComponents(
        first_layer_class=Qwen3_5DecoderLayer,
        decoder_layer_class=Qwen3_5DecoderLayer,
        norm_class=Qwen3_5RMSNorm,
        rotary_emb_class=Qwen3_5TextRotaryEmbedding,
    )

with suppress(ImportError):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    model_classes["qwen3_5_moe_text"] = ModelComponents(
        first_layer_class=Qwen3_5MoeDecoderLayer,
        decoder_layer_class=Qwen3_5MoeDecoderLayer,
        norm_class=Qwen3_5MoeRMSNorm,
        rotary_emb_class=Qwen3_5MoeTextRotaryEmbedding,
    )


def override_components(model_type: str, **overrides) -> ModelComponents:
    """Override specific components from a base model architecture.

    Used for speculative decoding to swap custom layers (typically first_layer_class)
    while inheriting other components from the base model.

    Args:
        model_type: Base model type key in ``model_classes``.
        **overrides: Component fields to override (first_layer_class,
            decoder_layer_class, etc).

    Returns:
        ModelComponents with specified overrides applied.
    """
    base = model_classes[model_type]
    return base._replace(**overrides)
