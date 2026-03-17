"""Lightweight model-config helpers shared across data-generation modules.

These utilities have **no vLLM dependency** and may be imported in any
environment (training, datagen, or test).
"""

from transformers import AutoConfig

__all__: list[str] = []  # internal — not part of the public API


def num_hidden_layers(model_path: str) -> int:
    """Return the number of transformer hidden layers for *model_path*.

    Handles both flat configs (``config.num_hidden_layers``) and nested
    multimodal configs (``config.text_config.num_hidden_layers``).

    :raises ValueError: if the layer count cannot be determined.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers  # type: ignore[return-value]
    if hasattr(config, "text_config"):
        return config.text_config.num_hidden_layers  # type: ignore[return-value]
    raise ValueError(
        f"Cannot determine num_hidden_layers from config for {model_path!r}. "
        "Expected config.num_hidden_layers or config.text_config.num_hidden_layers."
    )
