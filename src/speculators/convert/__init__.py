"""
Checkpoint conversion utilities for Speculators.

Provides tools to convert existing speculative decoding model checkpoints from external
research repositories (Eagle, HASS, FastMTP, etc.) into the standardized Speculators
format.

Supported Research Repositories:
    - Eagle v1, v2, and v3: https://github.com/SafeAILab/EAGLE
    - HASS: https://github.com/HArmonizedSS/HASS
    - FastMTP (Qwen3-Next / TencentBAC MiMo): https://arxiv.org/abs/2509.18362
"""

from .entrypoints import convert_model
from .fast_mtp import FastMTPConverter

__all__ = ["FastMTPConverter", "convert_model"]
