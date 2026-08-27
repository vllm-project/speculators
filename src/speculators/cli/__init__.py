"""
Speculators CLI — unified command-line interface for speculative decoding.

Commands are grouped into two panels:
- Pipeline: steps in the speculator training workflow
- Tools: standalone utilities
"""

from importlib.metadata import version as pkg_version

import typer

from speculators.cli.convert import convert
from speculators.cli.generate_offline_data import generate_offline_data
from speculators.cli.prepare_data import prepare_data
from speculators.cli.regenerate_responses import regenerate_responses
from speculators.cli.stitch import stitch_command
from speculators.cli.train import train_command

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


app.command(rich_help_panel="Pipeline")(prepare_data)
app.command(name="stitch-mtp", rich_help_panel="Pipeline")(stitch_command)
app.command(rich_help_panel="Pipeline")(generate_offline_data)
app.command(rich_help_panel="Pipeline")(regenerate_responses)
app.command(
    name="train",
    rich_help_panel="Pipeline",
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)(train_command)
app.command(rich_help_panel="Tools")(convert)
