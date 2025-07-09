"""
Main CLI entry point for speculators.
"""

import json
from importlib.metadata import version as pkg_version
from typing import Annotated, Any, Optional

import click
import typer

from speculators.convert import convert_model

# Create main app
app = typer.Typer(
    name="speculators",
    help="Speculators - Tools for speculative decoding with LLMs",
    add_completion=False,
    no_args_is_help=True,
)


# Add convert command
@app.command()
def convert(
    model: str,
    output_path: Optional[str] = None,
    config: Optional[str] = None,
    verifier: Optional[str] = None,
    verifier_attachment_mode: Annotated[
        str, typer.Option(click_type=click.Choice(["detached", "full", "train_only"]))
    ] = "detached",
    validate_device: Optional[str] = None,
    algorithm: Annotated[
        str, typer.Option(click_type=click.Choice(["auto", "eagle", "eagle2", "hass"]))
    ] = "auto",
    algorithm_kwargs: Annotated[
        Optional[dict[str, Any]], typer.Option(parser=json.loads)
    ] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
):
    convert_model(
        model=model,
        output_path=output_path,
        config=config,
        verifier=verifier,
        verifier_attachment_mode=verifier_attachment_mode,
        validate_device=validate_device,
        algorithm=algorithm,
        algorithm_kwargs=algorithm_kwargs,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
    )


@app.command()
def version():
    """Show the speculators version."""
    typer.echo(f"speculators version: {pkg_version('speculators')}")


if __name__ == "__main__":
    app()
