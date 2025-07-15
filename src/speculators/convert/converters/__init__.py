"""
Converter implementations for Speculators model format conversion.

This module provides converter classes for transforming external research model
checkpoints into the standardized Speculators format. The converters handle
architecture adaptation, configuration translation, and weight remapping for
various speculative decoding algorithms.

The module includes both the base converter interface and specific implementations
for different research repositories. All converters are registered automatically
through importing into the converters __init__.py module and can be accessed through
the base converter's registry system.

Classes:
    SpeculatorConverter: Abstract base class for all model converters with
        registry support
    EagleSpeculatorConverter: Converter for Eagle/HASS research repository
        checkpoints

Supported Research Repositories:
    - Eagle v1 and v2: https://github.com/SafeAILab/EAGLE
    - HASS: https://github.com/HArmonizedSS/HASS

Usage:
::
    from speculators.convert.converters import SpeculatorConverter

    # Get converter for specific algorithm
    converter = SpeculatorConverter.get_converter("eagle")

    # Convert model checkpoint
    config, model = converter.convert(
        model="path/to/checkpoint",
        output_path="converted_model"
    )
"""

from .base import SpeculatorConverter
from .eagle import EagleSpeculatorConverter

__all__ = [
    "EagleSpeculatorConverter",
    "SpeculatorConverter",
]
