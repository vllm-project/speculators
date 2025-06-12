from .independent import IndependentSpeculatorConfig
from .llama_eagle import LlamaEagleSpeculator, LlamaEagleSpeculatorConfig
from .mlp import MLPSpeculatorConfig
from .transformer import TransformerSpeculatorConfig

__all__ = [
    "IndependentSpeculatorConfig",
    "LlamaEagleSpeculator",
    "LlamaEagleSpeculatorConfig",
    "MLPSpeculatorConfig",
    "TransformerSpeculatorConfig",
]
