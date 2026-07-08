"""Medusa speculator implementation."""

from speculators.models.medusa.config import MedusaSpeculatorConfig
from speculators.models.medusa.core import MedusaDraftModel, compute_head_weights
from speculators.models.medusa.data import shift_batch_medusa

__all__ = [
    "MedusaDraftModel",
    "MedusaSpeculatorConfig",
    "compute_head_weights",
    "shift_batch_medusa",
]
