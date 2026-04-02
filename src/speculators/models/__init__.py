from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig

__all__ = [
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
]
