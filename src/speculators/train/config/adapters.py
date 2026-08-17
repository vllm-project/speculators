"""Adapters for released draft wrapper configurations."""

from collections.abc import Callable

from transformers import PretrainedConfig

_DraftConfigAdapter = Callable[[dict, PretrainedConfig], dict]


def _adapt_muse_glimmer_assistant(
    config: dict, verifier_config: PretrainedConfig
) -> dict:
    config = dict(config)
    config["model_type"] = "qwen3"
    config.setdefault("vocab_size", verifier_config.vocab_size)
    config.setdefault(
        "use_sliding_window",
        "sliding_attention" in config.get("layer_types", []),
    )
    for key in ("architectures", "block_size", "mask_token_id", "target_layer_ids"):
        config.pop(key, None)
    return config


_DRAFT_CONFIG_ADAPTERS: dict[str, _DraftConfigAdapter] = {
    "muse_glimmer_assistant": _adapt_muse_glimmer_assistant,
}


def adapt_draft_config(config: dict, verifier_config: PretrainedConfig) -> dict:
    """Normalize released draft wrapper configs to supported decoders."""
    model_type = config.get("model_type")
    adapter = (
        _DRAFT_CONFIG_ADAPTERS.get(model_type) if isinstance(model_type, str) else None
    )
    return adapter(config, verifier_config) if adapter else dict(config)
