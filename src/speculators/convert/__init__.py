"""
Checkpoint conversion utilities for Speculators.

This module provides tools to convert existing speculator checkpoints from external
research repositories (Eagle, HASS, etc.) into the standardized Speculators format.
The conversion process handles model architecture adaptation, configuration translation,
and optional verifier attachment for speculative decoding.

The primary entry point is the `convert_model` function, which supports automatic
algorithm detection and conversion from various input formats including local
checkpoints, Hugging Face model IDs, and PyTorch model instances.

Supported Research Repositories:
    - Eagle v1 and v2: https://github.com/SafeAILab/EAGLE
    - HASS: https://github.com/HArmonizedSS/HASS

Functions:
    convert_model: Convert external model checkpoints to Speculators-compatible format

Usage:
::
    from speculators.convert import convert_model

    # Convert with automatic algorithm detection
    model = convert_model("path/to/checkpoint", output_path="converted_model")

    # Convert with specific algorithm and verifier
    model = convert_model(
        model="hf_model_id",
        verifier="verifier_model_id",
        output_path="my_speculator"
    )
"""

from .converters import SpeculatorConverter
from .entrypoints import convert_model

__all__ = ["SpeculatorConverter", "convert_model"]
