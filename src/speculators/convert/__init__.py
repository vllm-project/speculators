"""
Checkpoint conversion utilities for Speculators.

This module provides tools to convert existing speculator checkpoints
(Eagle, HASS, etc.) into the standardized speculators format.
"""

from .entrypoints import convert_model

__all__ = ["convert_model"]
