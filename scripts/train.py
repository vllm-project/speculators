#!/usr/bin/env python3
"""Backward-compatibility shim — use ``speculators train`` instead.

For distributed training::

    torchrun --nproc_per_node=<N> -m speculators.train config.yaml
"""

import warnings

from speculators.train.cli import main
from speculators.train.config import TrainConfig

warnings.warn(
    "scripts/train.py is deprecated and will be removed in v0.9.0. "
    "Use 'speculators train' or 'torchrun -m speculators.train' instead.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    main(TrainConfig.resolve())
