from .eagle import EagleSpeculator, EagleSpeculatorConfig
from .eagle3 import Eagle3Speculator, Eagle3SpeculatorConfig
from .independent import IndependentSpeculatorConfig
from .mlp import MLPSpeculatorConfig

__all__ = [
    "Eagle3Speculator",
    "Eagle3SpeculatorConfig",
    "EagleSpeculator",
    "EagleSpeculatorConfig",
    "IndependentSpeculatorConfig",
    "MLPSpeculatorConfig",
]
