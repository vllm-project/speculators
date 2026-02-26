"""Architecture component registry for FastMTP models.

Defines reusable building blocks for different FastMTP architectures
(MIMO, Qwen3-Next, DeepSeek-V3), enabling architecture-agnostic code.
"""

from typing import NamedTuple

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
)

# Optional imports for Qwen3-Next
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextRMSNorm,
        Qwen3NextSparseMoeBlock,
    )

    HAS_QWEN3_NEXT = True
except ImportError:
    HAS_QWEN3_NEXT = False
    Qwen3NextAttention = None  # type: ignore[assignment,misc]
    Qwen3NextRMSNorm = None  # type: ignore[assignment,misc]
    Qwen3NextSparseMoeBlock = None  # type: ignore[assignment,misc]


__all__ = ["FastMTPComponents", "fast_mtp_components", "get_fast_mtp_components"]


class FastMTPComponents(NamedTuple):
    """Building blocks for a single FastMTP architecture.

    Different transformer architectures use different implementations of
    attention, MLP, and normalization. This tuple groups these components
    along with metadata about how they're stored in checkpoints.

    :param attention_class: Attention module (e.g., Qwen2Attention)
    :param mlp_class: Feedforward network module (standard MLP or MoE)
    :param norm_class: Normalization layer (e.g., Qwen2RMSNorm)
    :param checkpoint_key_prefix: Weight key prefix in saved checkpoints
    :param use_moe: Whether this architecture uses mixture-of-experts
    :param attention_bias: Whether attention projections include bias terms
    """

    attention_class: type
    mlp_class: type
    norm_class: type
    checkpoint_key_prefix: str
    use_moe: bool = False
    attention_bias: bool = False


fast_mtp_components: dict[str, FastMTPComponents] = {
    "mimo": FastMTPComponents(
        attention_class=Qwen2Attention,
        mlp_class=Qwen2MLP,
        norm_class=Qwen2RMSNorm,
        checkpoint_key_prefix="model.mtp_layers.0",
        use_moe=False,
        attention_bias=True,
    ),
}

# Add Qwen3-Next support if available
if HAS_QWEN3_NEXT:
    fast_mtp_components["qwen3_next"] = FastMTPComponents(
        attention_class=Qwen3NextAttention,
        mlp_class=Qwen3NextSparseMoeBlock,
        norm_class=Qwen3NextRMSNorm,
        checkpoint_key_prefix="mtp.layers.0",
        use_moe=True,
        attention_bias=False,  # Qwen3-Next typically doesn't use attention bias
    )


def get_fast_mtp_components(model_type: str) -> FastMTPComponents:
    """Retrieve architecture components for model_type.

    To add a new architecture: import its Attention, MLP, and RMSNorm classes,
    add an entry to the fast_mtp_components dict, and set checkpoint_key_prefix from
    actual checkpoint inspection.

    :param model_type: Transformer architecture name (e.g., "mimo", "qwen3_next")
    :return: Component bundle for the specified architecture
    :raises ValueError: If model_type is not registered
    :raises ImportError: If model_type requires optional dependencies not installed
    """
    if model_type not in fast_mtp_components:
        # Provide helpful error messages for known but unavailable model types
        if model_type in ["qwen3_next"] and not HAS_QWEN3_NEXT:
            raise ImportError(
                f"Model type '{model_type}' requires Qwen3-Next components "
                "which are not available. Please install a newer version of "
                "transformers or check your installation."
            )

        # Generic error for truly unsupported types
        supported = ", ".join(fast_mtp_components.keys())
        raise ValueError(
            f"Unsupported model_type '{model_type}' for FastMTP. Supported: {supported}"
        )
    return fast_mtp_components[model_type]
