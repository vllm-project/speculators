"""Reproducibility artifacts written next to the checkpoints.

Every run ships the exact configuration that produced it, so a checkpoint is
re-runnable without reconstructing flags by hand:

* ``run.yaml`` -- the fully-resolved, stage-shaped config; clean (no provenance
  inlined) so it round-trips straight back through ``--config``.
* ``train_command.txt`` -- the argv + git SHA + world size + library versions
  (written by :func:`speculators.train.utils.save_train_command`).
* ``run.provenance.yaml`` -- an audit-only sidecar recording which layer set each
  value. Never a ``--config`` input, so ``run.yaml`` stays re-runnable.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .schema import TrainConfig


def dump_yaml(cfg: "TrainConfig") -> str:
    """Serialize a resolved config to the stage-shaped, round-trippable YAML.

    Everything nests under a top-level ``train:`` key (the canonical form), so the
    file is forward-compatible with a future ``prepare_data:`` / ``launch_vllm:``
    stage and re-loads cleanly via ``--config``.
    """
    return yaml.safe_dump(
        {"train": cfg.model_dump(mode="json")},
        sort_keys=False,
        default_flow_style=False,
    )


def _dump_provenance(cfg: "TrainConfig") -> str:
    """Render the audit-only ``run.provenance.yaml`` body (with a header comment)."""
    header = (
        "# Audit-only provenance sidecar. Records which layer supplied each\n"
        "# resolved value (winner) and the full contributor trail, highest\n"
        "# precedence first: flag > set > yaml > default. NOT a --config input --\n"
        "# run.yaml stays clean and re-runnable.\n"
    )
    body = yaml.safe_dump(
        cfg.provenance.as_sidecar(),
        sort_keys=False,
        default_flow_style=False,
    )
    return header + body


def save(cfg: "TrainConfig", save_dir: str) -> None:
    """Write ``run.yaml`` + ``train_command.txt`` + ``run.provenance.yaml``.

    Called at rank 0 next to the checkpoints. Complements the resolved config with
    the environment manifest and the provenance audit trail.
    """
    # Local import: utils pulls in the (heavier) preprocessing stack, and keeping
    # it out of config import time avoids a needless cost + any import cycle.
    from speculators.train.utils import save_train_command  # noqa: PLC0415

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.yaml").write_text(dump_yaml(cfg))
    # Record the exact argv this config was resolved from (falls back to the live
    # sys.argv when the config was built off-argv, e.g. via from_flat). Reading the
    # config's own recorded argv here is friend access from its serialization seam.
    save_train_command(save_dir, command=cfg._argv)  # noqa: SLF001
    (out_dir / "run.provenance.yaml").write_text(_dump_provenance(cfg))
