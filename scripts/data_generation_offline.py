#!/usr/bin/env python3
"""Backward-compatibility shim — use ``speculators generate-data`` instead."""

import warnings

import typer

from speculators.cli.generate_data import generate_data

warnings.warn(
    "scripts/data_generation_offline.py is deprecated and will be removed in v0.9.0. "
    "Use 'speculators generate-data' instead.",
    DeprecationWarning,
    stacklevel=1,
)

app = typer.Typer()
app.command()(generate_data)

if __name__ == "__main__":
    app()
