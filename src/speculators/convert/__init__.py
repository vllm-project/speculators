"""
Checkpoint conversion utilities for Speculators.

This module provides tools to convert existing speculator checkpoints
(Eagle, HASS, etc.) into the standardized speculators format.
"""

from speculators.convert.eagle.eagle_converter import EagleConverter

__all__ = ["EagleConverter"]
