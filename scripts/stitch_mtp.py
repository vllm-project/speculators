#!/usr/bin/env python3
"""Backward-compatibility shim — use ``speculators stitch-mtp`` instead."""

import warnings

import typer

from speculators.cli.stitch import stitch_command

warnings.warn(
    "scripts/stitch_mtp.py is deprecated and will be removed in v0.9.0. "
    "Use 'speculators stitch-mtp' instead.",
    DeprecationWarning,
    stacklevel=1,
)

app = typer.Typer(rich_markup_mode="rich")
app.command()(stitch_command)

if __name__ == "__main__":
    app()
