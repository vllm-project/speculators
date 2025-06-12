"""Main CLI entry point for Speculators."""

import typer

from speculators.cli.convert import convert_command

app = typer.Typer(
    name="speculators",
    help="Speculators: A toolkit for speculative decoding in LLMs",
    no_args_is_help=True,
)

# Register commands
app.command(name="convert", help="Convert models to Speculators format")(convert_command)

if __name__ == "__main__":
    app()