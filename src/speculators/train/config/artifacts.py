"""Reproducibility artifacts: the ``run.yaml`` + ``train_command.txt`` writers.

``run.yaml`` is the resolved, provenance-free config that round-trips through
``--config``; ``train_command.txt`` records argv, git SHA, world size, versions.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .schema import nest_flat

if TYPE_CHECKING:
    from .schema import TrainConfig


def dump_yaml(cfg: "TrainConfig") -> str:
    """Serialize a resolved config to the stage-shaped, round-trippable YAML.

    Everything nests under a top-level ``train:`` key, so the file re-loads
    cleanly via ``--config`` and leaves room for a future ``prepare_data:`` /
    ``launch_vllm:`` stage.

    Only values a non-default layer supplied (flag or yaml) are emitted; defaults
    are omitted, which is what makes the file round-trip: the draft-init conflict
    check treats any YAML-supplied decoder-shaping key as explicit, so persisting
    every materialized default would make a ``--from-pretrained`` run.yaml reject
    itself on reload. Re-resolving the emitted subset re-derives the omitted
    defaults, yielding an identical config. The provenance record only selects
    which keys to emit; it is never inlined.
    """
    # Iterate the pinned flatten() order (not the provenance dict, whose key
    # order varies with how the config was populated) so the emitted YAML is
    # byte-stable across reloads regardless of flag-vs-yaml source.
    resolved = cfg.flatten()
    if cfg.provenance:
        provided = {
            dest: value
            for dest, value in resolved.items()
            if cfg.provenance[dest] != "default"
        }
    else:
        # A from_flat config carries no layer provenance, so "customized" is
        # defined as "differs from the default-constructed config".
        baseline = type(cfg)().flatten()
        provided = {
            dest: value
            for dest, value in resolved.items()
            if value != baseline.get(dest)
        }
    return yaml.safe_dump(
        {"train": nest_flat(provided)},
        sort_keys=False,
        default_flow_style=False,
    )


def save(cfg: "TrainConfig", save_dir: str) -> None:
    """Write ``run.yaml`` + ``train_command.txt`` next to the checkpoints.

    Called at rank 0 by ``scripts/train.py`` so every checkpoint carries the
    config that produced it. ``run.yaml`` is the clean resolved config (re-run via
    ``--config run.yaml``); ``train_command.txt`` records the exact argv this
    config was resolved from plus the environment manifest.
    """
    # Local import: utils pulls in the (heavier) preprocessing stack, so keeping
    # it out of config import time avoids a needless cost and any import cycle.
    from speculators.train.utils import save_train_command  # noqa: PLC0415

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.yaml").write_text(dump_yaml(cfg))
    # Record the argv this config was resolved from (falls back to the live
    # sys.argv when the config was built off-argv, e.g. via from_flat).
    save_train_command(save_dir, argv=list(cfg.argv))
