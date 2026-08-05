#!/usr/bin/env python3
"""Backward-compatibility shim — use ``speculators prepare-data`` instead."""

import warnings

import typer

from speculators.cli.prepare_data import prepare_data

warnings.warn(
    "scripts/prepare_data.py is deprecated. Use 'speculators prepare-data' instead.",
    DeprecationWarning,
    stacklevel=1,
)

app = typer.Typer()
app.command()(prepare_data)

if __name__ == "__main__":
    app()
