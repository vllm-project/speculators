"""MTP (Multi-Token Prediction) speculator implementation."""

from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.core import MTPDraftModel, compute_step_weights
from speculators.models.mtp.data import shift_batch_mtp

__all__ = [
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "compute_step_weights",
    "shift_batch_mtp",
]
