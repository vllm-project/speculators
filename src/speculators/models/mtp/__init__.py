"""MTP (Multi-Token Prediction) speculator implementation."""

from speculators.models.mtp.config import MTPConfig
from speculators.models.mtp.core import MTPDraftModel

__all__ = [
    "MTPConfig",
    "MTPDraftModel",
]
