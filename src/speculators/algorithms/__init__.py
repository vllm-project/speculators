from .eagle import EagleSpeculator
from .eagle2 import Eagle2Speculator
from .eagle3 import Eagle3Speculator
from .hass import HASSSpeculator
from .mlp_speculator import MLPSpeculator
from .specdec import SpecDecSpeculator

__all__ = [
    "Eagle2Speculator",
    "Eagle3Speculator",
    "EagleSpeculator",
    "HASSSpeculator",
    "MLPSpeculator",
    "SpecDecSpeculator",
]
