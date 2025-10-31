"""Data generation utilities for EAGLE-style speculative decoding training."""

from .config_generator import (
    DataGenerationConfig,
    extract_config_from_generator,
    generate_config,
)
from .vllm_hidden_states_generator import VllmHiddenStatesGenerator

__all__ = [
    "DataGenerationConfig",
    "VllmHiddenStatesGenerator",
    "extract_config_from_generator",
    "generate_config",
]
