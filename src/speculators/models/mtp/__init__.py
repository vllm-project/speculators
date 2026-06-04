"""MTP (Multi-Token Prediction) speculator implementation."""

from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.core import MTPDraftModel, compute_step_weights

__all__ = [
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "compute_step_weights",
]
