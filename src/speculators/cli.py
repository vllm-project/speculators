"""
Main CLI entry point for speculators.
"""

from importlib.metadata import version as pkg_version
from typing import Optional

import typer  # type: ignore[import-not-found]

from speculators.convert.cli import convert


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        typer.echo(f"speculators version: {pkg_version('speculators')}")
        raise typer.Exit


# Create main app
app = typer.Typer(
    name="speculators",
    help="Speculators - Tools for speculative decoding with LLMs",
    add_completion=False,
    no_args_is_help=True,
)

# Add convert command
app.command(name="convert", help="Convert checkpoints to speculators format")(convert)


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the speculators version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    Speculators - Tools for speculative decoding with LLMs.
    """


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
