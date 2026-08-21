"""
Speculators CLI — unified command-line interface for speculative decoding.

Commands are grouped into two panels:
- Pipeline: steps in the speculator training workflow
- Tools: standalone utilities
"""

from importlib.metadata import version as pkg_version

import typer

from speculators.cli.convert import convert
from speculators.cli.regenerate_responses import regenerate_responses

__all__ = ["app"]

app = typer.Typer(
    name="speculators",
    help="Speculators - speculative decoding for vLLM",
    no_args_is_help=True,
)


def _version_callback(value: bool):
    if value:
        typer.echo(f"speculators version: {pkg_version('speculators')}")
        raise typer.Exit


@app.callback()
def _main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
    ),
):
    pass


app.command(rich_help_panel="Pipeline")(regenerate_responses)
app.command(rich_help_panel="Tools")(convert)
