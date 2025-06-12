"""Converters for transforming models to Speculators format."""

from speculators.converters.base import BaseConverter, ConversionResult
from speculators.converters.llama_eagle import LlamaEagleConverter

__all__ = [
    "BaseConverter",
    "ConversionResult",
    "LlamaEagleConverter",
]