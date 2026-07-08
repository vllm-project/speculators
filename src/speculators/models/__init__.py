from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .dspark import DSparkDraftModel, DSparkSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .hydra import HydraDraftModel, HydraSpeculatorConfig
from .mtp import MTPDraftModel, MTPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig

__all__ = [
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "DSparkDraftModel",
    "DSparkSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "HydraDraftModel",
    "HydraSpeculatorConfig",
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
]
