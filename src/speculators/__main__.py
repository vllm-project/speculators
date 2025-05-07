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


if __name__ == "__main__":
    cli()
