"""
Main CLI entry point for speculators.
"""

from importlib.metadata import version as pkg_version

import typer

from speculators.convert.__main__ import convert

# Create main app
app = typer.Typer(
    name="speculators",
    help="Speculators - Tools for speculative decoding with LLMs",
    add_completion=False,
    no_args_is_help=True,
)

# Add convert command
app.command(name="convert", help="Convert checkpoints to speculators format")(convert)


@app.command()
def version():
    """Show the speculators version."""
    typer.echo(f"speculators version: {pkg_version('speculators')}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
