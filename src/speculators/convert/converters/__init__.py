"""
Registry-based converter architecture for transforming external checkpoints.

This module provides the converter framework for standardizing external research model
checkpoints into the Speculators format. The converter system uses a registry pattern
to automatically detect and instantiate appropriate converters based on algorithm type
and model characteristics, supporting extensible conversion workflows with validation.
"""

from __future__ import annotations

from .base import SpeculatorConverter

__all__ = ["SpeculatorConverter"]
