#!/usr/bin/env python3
"""Backward-compatibility shim — use ``speculators generate-offline-data`` instead."""

import warnings

import typer

from speculators.cli.generate_offline_data import generate_offline_data

warnings.warn(
    "scripts/data_generation_offline.py is deprecated and will be removed in v0.9.0. "
    "Use 'speculators generate-offline-data' instead.",
    DeprecationWarning,
    stacklevel=1,
)

app = typer.Typer()
app.command()(generate_offline_data)

if __name__ == "__main__":
    app()
