from .eagle import EagleSpeculator, EagleSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .independent import IndependentSpeculatorConfig
from .mlp import MLPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig

__all__ = [
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "EagleSpeculator",
    "EagleSpeculatorConfig",
    "IndependentSpeculatorConfig",
    "MLPSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
]
