"""
Checkpoint conversion utilities for Speculators.

Provides tools to convert existing speculative decoding model checkpoints from external
research repositories (Eagle, HASS, etc.) into the standardized Speculators format.

Supported Research Repositories:
    - Eagle v1, v2, and v3: https://github.com/SafeAILab/EAGLE
    - HASS: https://github.com/HArmonizedSS/HASS
"""

from .entrypoints import SUPPORTED_ALGORITHMS, convert_model

__all__ = ["SUPPORTED_ALGORITHMS", "convert_model"]
