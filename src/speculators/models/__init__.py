from .eagle import EagleSpeculatorConfig
from .independent import IndependentSpeculatorConfig
from .llama_eagle import LlamaEagleSpeculator, LlamaEagleSpeculatorConfig
from .mlp import MLPSpeculatorConfig

__all__ = [
    "EagleSpeculatorConfig",
    "IndependentSpeculatorConfig",
    "LlamaEagleSpeculator",
    "LlamaEagleSpeculatorConfig",
    "MLPSpeculatorConfig",
]
