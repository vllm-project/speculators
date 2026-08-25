from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .dspark import DSparkDraftModel, DSparkSpeculatorConfig
from .dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .mtp import MTPDraftModel, MTPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig

__all__ = [
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "DSV4DSparkConfig",
    "DSV4DSparkDraftModel",
    "DSparkDraftModel",
    "DSparkSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
]
