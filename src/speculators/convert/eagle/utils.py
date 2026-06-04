"""
Eagle-specific utility functions for checkpoint conversion.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from loguru import logger
from transformers import LlamaConfig

_LLAMA_CONFIG_PARAMS = set(inspect.signature(LlamaConfig.__init__).parameters)
_LLAMA_CONFIG_HAS_ROPE_THETA = "rope_theta" in _LLAMA_CONFIG_PARAMS
_LLAMA_CONFIG_HAS_TORCH_DTYPE = "torch_dtype" in _LLAMA_CONFIG_PARAMS


def build_llama_config_dtype_kwarg(torch_dtype: str | None) -> dict[str, Any]:
    if torch_dtype is None:
        return {}
    if _LLAMA_CONFIG_HAS_TORCH_DTYPE:
        return {"torch_dtype": torch_dtype}
    return {"dtype": torch_dtype}


def build_llama_config_rope_kwargs(
    rope_theta: float = 10000.0,
    rope_scaling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _LLAMA_CONFIG_HAS_ROPE_THETA:
        kwargs: dict[str, Any] = {"rope_theta": rope_theta}
        if rope_scaling is not None:
            kwargs["rope_scaling"] = rope_scaling
        return kwargs

    rope_params: dict[str, Any] = {"rope_theta": rope_theta}
    if rope_scaling is not None:
        rope_params["rope_type"] = rope_scaling.get(
            "rope_type", rope_scaling.get("type", "default")
        )
        for k, v in rope_scaling.items():
            if k not in ("rope_type", "type"):
                rope_params[k] = v
    else:
        rope_params["rope_type"] = "default"
    return {"rope_parameters": rope_params}


def find_vocab_size(config_dict: dict) -> int | None:
    """
    Recursively search for vocab_size in nested config dictionary.

    :param config_dict: Configuration dictionary to search
    :return: vocab_size if found, None otherwise
    """
    if isinstance(config_dict, dict):
        if "vocab_size" in config_dict:
            return config_dict["vocab_size"]
        for value in config_dict.values():
            if isinstance(value, dict):
                result = find_vocab_size(value)
                if result is not None:
                    return result
    return None


def detect_fusion_bias_and_layernorms(
    weights: dict[str, torch.Tensor],
) -> tuple[bool, bool]:
    """
    Auto-detect fusion bias and extra layernorms presence based on weight names.

    :param weights: Dictionary of weight tensors
    :return: Tuple of (has_fusion_bias, has_layernorms)

    :Example:

        >>> weights = {
        ...     "fc.bias": torch.randn(4096),
        ...     "embed_layernorm.weight": torch.randn(4096)
        ... }
        >>> has_bias, has_ln = detect_fusion_bias_and_layernorms(weights)
        >>> print(f"Fusion bias: {has_bias}, Layernorms: {has_ln}")
        Fusion bias: True, Layernorms: True
    """
    has_fusion_bias = "fc.bias" in weights
    has_layernorms = any(
        name in weights
        for name in ["embed_layernorm.weight", "post_embedding_layernorm.weight"]
    )

    if has_fusion_bias:
        logger.info("Detected fusion bias in checkpoint")
    if has_layernorms:
        logger.info("Detected extra layernorms in checkpoint")

    return has_fusion_bias, has_layernorms
