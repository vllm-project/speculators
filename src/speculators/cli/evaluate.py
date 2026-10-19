"""Expose the source-checkout evaluator through the unified CLI."""

import os
import sys
from pathlib import Path

import typer


def evaluate_command(ctx: typer.Context) -> None:
    """Evaluate acceptance or speedup using scripts/evaluate/evaluate.py."""
    script = (
        Path(__file__).resolve().parents[3] / "scripts" / "evaluate" / "evaluate.py"
    )
    os.execv(sys.executable, [sys.executable, str(script), *ctx.args])  # noqa: S606
