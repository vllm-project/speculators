"""DeepSeek-V4-Flash DSpark draft speculator.

A self-contained, backend-agnostic implementation of the DSpark draft *backbone* for
DeepSeek-V4-Flash: multi-head latent attention with per-head attention sinks, 256 routed
experts + 1 shared per layer, and hyper-connections in place of the residual stream --
plain PyTorch throughout.

The DSpark *method* (anchor-block sampling, the Markov and confidence heads, the
compound loss, and the SpeculatorModel / trainer / data contract) is reused as-is from
:mod:`speculators.models.dspark`; this package adds only the DSV4-native backbone and
the thin subclass that swaps it in.
"""

from __future__ import annotations

from .config import DSparkDraftConfig
from .core import DSV4DSparkConfig, DSV4DSparkDraftModel

__all__ = ["DSV4DSparkConfig", "DSV4DSparkDraftModel", "DSparkDraftConfig"]
