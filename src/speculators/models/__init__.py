from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from .dspark import DSparkDraftModel, DSparkSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .modular import ModularDraftModel, ModularSpeculatorConfig
from .mtp import MTPDraftModel, MTPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig
from .xpress import XPressDraftModel, XPressSpeculatorConfig

__all__ = [
    "DFlash2DraftModel",
    "DFlash2SpeculatorConfig",
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "DSparkDraftModel",
    "DSparkSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "ModularDraftModel",
    "ModularSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
    "XPressDraftModel",
    "XPressSpeculatorConfig",
]
