from .eagle import EagleSpeculator, EagleSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .fastmtp import FastMTPDraftModel, FastMTPSpeculatorConfig
from .independent import IndependentSpeculatorConfig
from .mlp import MLPSpeculatorConfig

__all__ = [
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "EagleSpeculator",
    "EagleSpeculatorConfig",
    "FastMTPDraftModel",
    "FastMTPSpeculatorConfig",
    "IndependentSpeculatorConfig",
    "MLPSpeculatorConfig",
]
