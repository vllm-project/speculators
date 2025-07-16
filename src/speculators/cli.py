"""
Main CLI entry point for Speculators.
Supports checkpoint conversion for Eagle and Eagle-3 models.
"""

from importlib.metadata import version as pkg_version
from typing import Annotated, Optional

import typer  # type: ignore[import-not-found]

from speculators.convert.eagle.eagle_converter import EagleConverter
from speculators.convert.eagle.eagle3_converter import Eagle3Converter

app = typer.Typer(
    name="speculators",
    help="Speculators - Tools for speculative decoding with LLMs",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def version_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the speculators version and exit",
        callback=lambda v: (
            typer.echo(f"speculators version: {pkg_version('speculators')}")
            or raise_exit()
        ) if v else None,
        is_eager=True,
    ),
):
    """
    Speculators - Tools for speculative decoding with LLMs.
    """


def raise_exit():
    raise typer.Exit()


@app.command(name="convert")
def convert(
    input_path: Annotated[
        str,
        typer.Argument(help="Path to checkpoint (local path or HuggingFace model ID)"),
    ],
    output_path: Annotated[
        str,
        typer.Argument(help="Output directory for the converted checkpoint"),
    ],
    base_model: Annotated[
        str,
        typer.Argument(help="Base model name/path (e.g., meta-llama/Llama-3.1-8B)"),
    ],
    # Model type flags (mutually exclusive)
    eagle: Annotated[
        bool,
        typer.Option("--eagle", help="Convert Eagle/HASS checkpoint"),
    ] = False,
    eagle3: Annotated[
        bool,
        typer.Option("--eagle3", help="Convert Eagle-3 checkpoint"),
    ] = False,
    # Model-specific options
    layernorms: Annotated[
        bool,
        typer.Option("--layernorms", help="Enable extra layernorms (Eagle/HASS only)"),
    ] = False,
    fusion_bias: Annotated[
        bool,
        typer.Option("--fusion-bias", help="Enable fusion bias (Eagle/HASS only)"),
    ] = False,
    # General options
    validate: Annotated[
        bool,
        typer.Option("--validate/--no-validate", help="Validate the converted checkpoint"),
    ] = False,
):
    """
    Convert speculator checkpoints to the standardized speculators format.

    Examples:
        # Convert Eagle checkpoint
        speculators convert --eagle yuhuili/EAGLE-LLaMA3.1-Instruct-8B \\
            ./eagle-converted meta-llama/Llama-3.1-8B-Instruct

        # Convert Eagle with layernorms enabled
        speculators convert --eagle nm-testing/Eagle_TTT ./ttt-converted \\
            meta-llama/Llama-3.1-8B-Instruct --layernorms

        # Convert Eagle-3 checkpoint
        speculators convert --eagle3 nm-testing/SpeculatorLlama3-1-8B-Eagle3\\
            ./eagle3-converted meta-llama/Meta-Llama-3.1-8B-Instruct --validate
    """
    if sum([eagle, eagle3]) > 1:
        typer.echo("✗ Error: --eagle and --eagle3 are mutually exclusive.", err=True)
        raise typer.Exit(1)

    try:
        if eagle:
            converter = EagleConverter(
                model=input_path,
                verifier=base_model,
                output_path=output_path,
            )
            converter.convert(
                fusion_bias=fusion_bias,
                layernorms=layernorms,
                validate=validate,
            )
        elif eagle3:
            converter = Eagle3Converter(
                model=input_path,
                verifier=base_model,
                output_path=output_path,
            )
            converter.convert(
                validate=validate,
            )
        else:
            typer.echo("✗ Error: Specify one model type: --eagle or --eagle3", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"✗ Conversion failed: {e}", err=True)
        raise typer.Exit(1) from e


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

