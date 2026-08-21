#!/usr/bin/env python3
"""Backward-compatibility shim — use ``speculators regenerate-responses`` instead."""

import warnings

import typer

from speculators.cli.regenerate_responses import regenerate_responses

warnings.warn(
    "scripts/response_regeneration/script.py is deprecated and will be "
    "removed in v0.9.0. Use 'speculators regenerate-responses' instead.",
    DeprecationWarning,
    stacklevel=1,
)

app = typer.Typer()
app.command()(regenerate_responses)

if __name__ == "__main__":
    app()
