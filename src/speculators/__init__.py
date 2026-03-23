"""
Speculators: A Unified Library for Speculative Decoding Algorithms for LLMs

Speculators provides a standardized framework for creating, representing, and
storing speculative decoding algorithms for large language model (LLM) inference.
It enables developers to implement and productize various speculative decoding
approaches with a consistent interface, making them ready for integration with
LLM inference servers like vLLM.

Speculative decoding is a technique that can significantly improve LLM inference
performance by predicting multiple tokens with a smaller, speculative model and
then verifying the predictions with the original, larger model.
This approach tradesoff extra computation for reduced latency, making it suitable
for real-time applications on deployments that are not compute-constrained.

The library offers a modular architecture with components for:
- Standardized interfaces for working with speculative decoding algorithms that
  build on top of Transformers pathways for simple integration.
- Centralized definition, configuration, and validation of speculative decoding
  algorithms.
"""

from .config import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    reload_schemas,
)
from .model import SpeculatorModel
from .models import Eagle3DraftModel, Eagle3SpeculatorConfig
from .proposals import TokenProposalConfig

__all__ = [
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "SpeculatorModel",
    "SpeculatorModelConfig",
    "SpeculatorsConfig",
    "TokenProposalConfig",
    "VerifierConfig",
    "reload_schemas",
]

# base imports complete, run auto loading for base classes
reload_schemas()
