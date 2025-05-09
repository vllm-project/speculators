"""
__main__.py

This module provides a CLI for speculators using the `click` library.
The CLI enables interacting with the library's functionality from the command line.
The root command `speculators` is the entrypoint, with subcommands for specific actions.

Subcommands:
- `config`: Prints the current configuration settings.

Example Usage:
```bash
speculators config
```
"""

from pathlib import Path
from typing import Optional, Union

import click

from speculators.settings import print_config


@click.group()
def cli():
    """
    The root command for the speculators CLI.
    """


@cli.command()
def config():
    """
    Print the current configuration settings.
    """
    print_config()


@cli.command()
def convert(
    source: Union[str, Path],  # noqa: ARG001
    config: Optional[Union[str, Path]] = None,  # noqa: ARG001
    output: Optional[Union[str, Path]] = None,  # noqa: ARG001
):
    """
    Convert a model from a specific format to the speculators library format.
    """
    raise NotImplementedError(
        "Model conversion is not yet implemented. "
        "Please provide a valid source and config."
    )


@cli.command()
def inference(**kwargs):  # noqa: ARG001
    """
    Run inference using the speculators library.
    """
    raise NotImplementedError(
        "Inference is not yet implemented. "
        "Please provide the necessary arguments for inference."
    )


@cli.command()
def validate(**kwargs):  # noqa: ARG001
    """
    Validate a model using the speculators library.
    """


@cli.command()
def train(**kwargs):  # noqa: ARG001
    """
    Train a model using the speculators library.
    """


if __name__ == "__main__":
    cli()
