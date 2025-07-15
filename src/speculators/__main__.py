"""
CLI entrypoints for the Speculators library.

This module provides a command-line interface for creating and managing speculative
decoding models. The CLI is built using Typer and provides commands for model
conversion, version information, and other utilities.

The CLI can be accessed through the `speculators` command after installation, or by
running this module directly with `python -m speculators`.

Commands:
    convert: Convert models from external repos/formats to supported Speculators models
    version: Display the current version of the Speculators library

Usage:
    $ speculators --help
    $ speculators --version
    $ speculators convert <model> [OPTIONS]
"""

import json
from importlib.metadata import version as pkg_version
from typing import Annotated, Any, Optional

import click
import typer

from speculators.convert import convert_model

__all__ = ["app"]

# Configure the main Typer application
app = typer.Typer(
    name="speculators",
    help="Speculators - Tools for speculative decoding with LLMs",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool):
    """
    Callback function to print the version of the Speculators package and exit.

    This function is used as a callback for the --version option in the main CLI.
    When the version option is specified, it prints the version information and
    exits the application.

    :param value: Boolean indicating whether the version option was specified.
        If True, prints version and exits.
    """
    if value:
        typer.echo(f"speculators version: {pkg_version('speculators')}")
        raise typer.Exit


@app.callback()
def speculators(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
    ),
):
    """
    Main entry point for the Speculators CLI application.

    This function serves as the root command callback and handles global options
    such as version display. It is automatically called by Typer when the CLI
    is invoked.

    :param ctx: The Typer context object containing runtime information.
    :param version: Boolean option to display version information and exit.
    """


# Add convert command
@app.command()
def convert(
    model: str,
    output_path: str = "speculators_converted",
    config: Optional[str] = None,
    verifier: Optional[str] = None,
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
    """
    Convert external models to Speculators-compatible format.

    This command converts models from external research repositories or formats
    into the standardized Speculators format. Currently supports model formats
    from the list of research repositories below with automatic algorithm detection.

    Supported Research Repositories:
        - Eagle v1 and v2: https://github.com/SafeAILab/EAGLE
        - HASS: https://github.com/HArmonizedSS/HASS

    :param model: Path to model checkpoint or Hugging Face model ID to convert.
    :param output_path: Directory path where converted model will be saved.
    :param config: Path to config.json file or HF model ID for model configuration.
        If not provided, configuration will be inferred from the checkpoint.
    :param verifier: Path to verifier checkpoint or HF model ID to attach as the
        verification model for speculative decoding.
    :param validate_device: Device identifier (e.g., "cpu", "cuda") for post-conversion
        validation. If not provided, validation is skipped.
    :param algorithm: Conversion algorithm to use. "auto" enables automatic detection
        based on model type and configuration.
    :param algorithm_kwargs: Additional keyword arguments for the conversion algorithm
        as a JSON string. Passed directly to the converter class.
    :param cache_dir: Directory for caching downloaded models. Uses default HF cache
        if not specified.
    :param force_download: Force re-download of checkpoint and config files,
        bypassing cache.
    :param local_files_only: Use only local files without attempting downloads
        from Hugging Face Hub.
    :param token: Hugging Face authentication token for accessing private models.
    :param revision: Git revision (branch, tag, or commit hash) for model files
        from Hugging Face Hub.
    """
    convert_model(
        model=model,
        output_path=output_path,
        config=config,
        verifier=verifier,
        validate_device=validate_device,
        algorithm=algorithm,  # type: ignore[arg-type]
        algorithm_kwargs=algorithm_kwargs,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
    )


if __name__ == "__main__":
    app()
