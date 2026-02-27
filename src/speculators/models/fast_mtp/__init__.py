"""FastMTP (Multi-Token Prediction) speculator implementation."""

from speculators.models.fast_mtp.config import FastMTPConfig
from speculators.models.fast_mtp.core import FastMTPLayer, FastMTPSpeculator
from speculators.models.fast_mtp.model_definitions import (
    FastMTPComponents,
    fast_mtp_components,
    get_fast_mtp_components,
)

__all__ = [
    "FastMTPComponents",
    "FastMTPConfig",
    "FastMTPLayer",
    "FastMTPSpeculator",
    "fast_mtp_components",
    "get_fast_mtp_components",
]
