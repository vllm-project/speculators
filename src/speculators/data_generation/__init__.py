"""Data generation utilities for speculative decoding training."""

from speculators.data_generation.fast_mtp_generator import (
    generate_and_save_fast_mtp,
)
from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
)

__all__ = ["VllmHiddenStatesGenerator", "generate_and_save_fast_mtp"]
