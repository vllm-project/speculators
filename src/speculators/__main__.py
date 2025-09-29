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
import typer  # type: ignore[import-not-found]

from speculators.convert import convert_model

__all__ = ["app"]

app = typer.Typer(
    name="speculators",
    help="Speculators - A unified library for speculative decoding algorithms for LLMs",
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


@app.command()
def convert(
    model: Annotated[
        str, typer.Argument(help="Model checkpoint or Hugging Face model ID to convert")
    ],
    verifier: Annotated[
        str,
        typer.Option(
            "--verifier",
            help=(
                "Verifier model checkpoint or Hugging Face model ID "
                "to attach as the verification/base model for speculative decoding"
            ),
        ),
    ],
    algorithm: Annotated[
        str,
        typer.Option(
            help=(
                "The source repo/algorithm to convert from into the matching algorithm "
                "in Speculators"
            ),
            click_type=click.Choice(["eagle", "eagle3"]),
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Directory path where converted model will be saved")
    ] = "converted",
    validate_device: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Device to validate the model on (e.g. 'cuda:0') "
                "If not provided, validation is skipped."
            ),
        ),
    ] = None,
    algorithm_kwargs: Annotated[
        Optional[dict[str, Any]],
        typer.Option(
            parser=json.loads,
            help=(
                "Additional keyword args for the conversion alg as a JSON string. "
                'Options for Eagle: {"layernorms": true, "fusion_bias": true}. '
                'Options for Eagle3: {"norm_before_residual": true, "inference_type": "text", "eagle_aux_hidden_state_layer_ids": [1,23,44]}.'
            ),
        ),
    ] = None,
):
    """
    Convert models from external research repositories or formats
    into the standardized Speculators format for use within the Speculators
    framework, Hugging Face model hub compatability, and deployment with vLLM.
    Supported algorithms, repositories, and examples given below.

    \b
    algorithm=="eagle":
        Eagle v1, v2: https://github.com/SafeAILab/EAGLE
        HASS: https://github.com/HArmonizedSS/HASS
        ::
        # general
        speculators convert "yuhuili/EAGLE-LLaMA3.1-Instruct-8B" \\
            --algorithm eagle \\
            --verifier "meta-llama/Llama-3.1-8B-Instruct"
        # with layernorms and fusion bias enabled
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle \\
            --algorithm-kwargs '{"layernorms": true, "fusion_bias": true}' \\
            --verifier "meta-llama/Llama-3.1-8B-Instruct"

    \b
    algorithm=="eagle3":
        Eagle v3: https://github.com/SafeAILab/EAGLE
        ::
        # general
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle3
            --verifier "meta-llama/Llama-3.1-8B-Instruct"
        # with normalization before the residual
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle3
            --algorithm-kwargs '{"norm_before_residual": true}'
            --verifier "meta-llama/Llama-3.1-8B-Instruct"
    """
    if not algorithm_kwargs:
        algorithm_kwargs = {}

    convert_model(
        model=model,
        verifier=verifier,
        output_path=output_path,
        validate_device=validate_device,
        algorithm=algorithm,  # type: ignore[arg-type]
        **algorithm_kwargs,
    )


if __name__ == "__main__":
    app()
