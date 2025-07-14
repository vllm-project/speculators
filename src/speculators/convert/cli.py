"""
Unified CLI interface for checkpoint conversion.
"""

from typing import Annotated

import typer  # type: ignore[import-not-found]

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.eagle.eagle_converter import EagleConverter

app = typer.Typer(
    help="Convert speculator checkpoints to the standardized speculators format.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
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
    eagle: Annotated[
        bool,
        typer.Option(
            "--eagle",
            help="Convert Eagle/HASS checkpoint",
        ),
    ] = False,
    eagle3: Annotated[
        bool,
        typer.Option(
            "--eagle3",
            help="Convert Eagle-3 checkpoint",
        ),
    ] = False,
    layernorms: Annotated[
        bool,
        typer.Option(
            "--layernorms",
            help="Enable extra layernorms (Eagle/HASS only)",
        ),
    ] = False,
    fusion_bias: Annotated[
        bool,
        typer.Option(
            "--fusion-bias",
            help="Enable fusion bias (Eagle/HASS only)",
        ),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate the converted checkpoint",
        ),
    ] = False,
):
    """
    Convert speculator checkpoints to speculators format.
    """
    if sum([eagle, eagle3]) > 1:
        typer.echo("Error: --eagle and --eagle3 are mutually exclusive.", err=True)
        raise typer.Exit(1)

    if eagle:
        converter = EagleConverter()
        try:
            converter.convert(
                input_path,
                output_path,
                base_model,
                fusion_bias=fusion_bias,
                layernorms=layernorms,
                validate=validate,
            )
        except Exception as e:
            typer.echo(f"✗ Conversion failed: {e}", err=True)
            raise typer.Exit(1) from e
    elif eagle3:
        converter = Eagle3Converter()  # type: ignore[assignment]
        try:
            converter.convert(
                input_path,
                output_path,
                base_model,
                validate=validate,
            )
        except Exception as e:
            typer.echo(f"✗ Conversion failed: {e}", err=True)
            raise typer.Exit(1) from e
    else:
        typer.echo("Error: Specify one model type: --eagle or --eagle3", err=True)
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
