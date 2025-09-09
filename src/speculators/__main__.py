"""
CLI entrypoints for the Speculators library.

This module provides a command-line interface for creating and managing speculative
decoding models. The CLI is built using Typer and provides commands for model
conversion, version information, and other utilities. It serves as the primary
entry point for users to interact with the Speculators library from the command line.

Example:
::
    speculators --help
    speculators convert "yuhuili/EAGLE-LLaMA3.1-Instruct-8B" \
        --algorithm eagle \
        --verifier "meta-llama/Llama-3.1-8B-Instruct"
"""

from __future__ import annotations

import json
from importlib.metadata import version as pkg_version
from typing import Annotated, Any, Literal, cast

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


def version_callback(value: bool) -> None:
    """
    Print the Speculators package version and exit.

    :param value: Whether the version option was specified
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

    :param ctx: Typer context object containing runtime information
    :param version: Option to display version information and exit
    """


@app.command()
def convert(
    model: Annotated[
        str, typer.Argument(help="Model checkpoint or Hugging Face model ID to convert")
    ],
    output_path: Annotated[
        str, typer.Option(help="Directory path where converted model will be saved")
    ] = "converted",
    config: str | None = None,
    verifier: Annotated[
        str | None,
        typer.Option(
            "--verifier",
            help=(
                "Verifier model checkpoint or Hugging Face model ID "
                "to attach as the verification/base model for speculative decoding"
            ),
        ),
    ] = None,
    validate_device: Annotated[
        str | None,
        typer.Option(
            help=(
                "Device to validate the model on (e.g. 'cuda:0') "
                "If not provided, validation is skipped."
            ),
        ),
    ] = None,
    algorithm: Annotated[
        str,
        typer.Option(
            help=(
                "The source repo/algorithm to convert from into the matching algorithm "
                "in Speculators"
            ),
            click_type=click.Choice(["auto", "eagle", "eagle2", "hass"]),
        ),
    ] = "auto",
    algorithm_kwargs: Annotated[
        dict[str, Any] | None,
        typer.Option(
            parser=json.loads,
            help=(
                "Additional keyword args for the conversion alg as a JSON string. "
                'Options for Eagle: {"layernorms": true, "fusion_bias": true}. '
                'Options for Eagle3: {"norm_before_residual": true}.'
            ),
        ),
    ] = None,
    cache_dir: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | None = None,
    revision: str | None = None,
) -> None:
    """
    Convert models from external research repositories into Speculators format.

    Converts models from research implementations (EAGLE, HASS) into standardized
    Speculators format for use with Hugging Face, vLLM, and the Speculators framework.

    [EAGLE v1, v2](https://github.com/SafeAILab/EAGLE),
    and [HASS](https://github.com/HArmonizedSS/HASS) Example:
    ::
        speculators convert "yuhuili/EAGLE-LLaMA3.1-Instruct-8B" \
            --verifier "meta-llama/Llama-3.1-8B-Instruct"

        # with layernorms and fusion bias enabled
        speculators convert "./eagle/checkpoint" \
            --algorithm-kwargs '{"layernorms": true, "fusion_bias": true}' \
            --verifier "meta-llama/Llama-3.1-8B-Instruct"

        # eagle3 with normalization before the residual
            --algorithm-kwargs '{"norm_before_residual": true}' \
            --verifier "meta-llama/Llama-3.1-8B-Instruct"

    :param model: Model checkpoint path or Hugging Face model ID to convert
    :param output_path: Directory path where converted model will be saved
    :param config: Optional config path, model ID, or config instance
    :param verifier: Optional verifier model for speculative decoding
    :param validate_device: Optional device for post-conversion validation
    :param algorithm: Source algorithm to convert from (auto, eagle, eagle2, hass)
    :param algorithm_kwargs: Additional conversion algorithm keyword arguments
    :param cache_dir: Optional directory for caching downloaded model files
    :param force_download: Force re-downloading files even if cached
    :param local_files_only: Use only local files without downloading from hub
    :param token: Optional Hugging Face authentication token for private models
    :param revision: Optional Git revision for downloading from Hugging Face hub
    """
    convert_model(
        model=model,
        output_path=output_path,
        config=config,
        verifier=verifier,
        validate_device=validate_device,
        algorithm=cast('Literal["auto", "eagle", "eagle2", "hass"]', algorithm),
        algorithm_kwargs=algorithm_kwargs or {},
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
    )


if __name__ == "__main__":
    app()
