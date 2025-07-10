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
    Convert a model from an external repo/format to a supported Speculators model.
    Currently supports conversion of Eagle, Eagle2, and HASS research repo models.

    :param model: Path to the model checkpoint or Hugging Face model ID.
    :param output_path: Path to save the converted Speculators model.
        Defaults to "speculators_converted" in the current directory.
    :param config: Optional path to a local config.json file or a Hugging Face model ID
        to use for the model configuration. If not provided, the model's config will be
        inferred from the checkpoint.
    :param verifier: Optional path to a verifier checkpoint or a Hugging Face model ID
        to attach to the converted Speculators model as the larger model the speculator
        will use to verify its predictions.
        If not provided, no verifier will be attached.
    :param validate_device: Optional device to validate the model on after conversion.
        Can be set to a string like "cpu", "cuda", or a specific device ID.
        If provided, the model will be validated on this device after conversion.
        If not provided, no validation will be performed.
    :param algorithm: The conversion algorithm to use.
        Can be "auto", "eagle", "eagle2", or "hass".
        Defaults to "auto", which will automatically select the appropriate algorithm
        based on the model type and configuration, if possible.
    :param algorithm_kwargs: Optional additional keyword arguments for the conversion
        algorithm. These will be passed directly to the converter class.
    :param cache_dir: Optional directory to cache downloaded models.
        If not provided, the default Hugging Face cache directory will be used.
    :param force_download: If True, forces redownload of the checkpoint and config.
        If False, will use cached versions if available.
    :param local_files_only: If True, only uses local files and does not attempt to
        download from the Hugging Face Hub.
    :param token: Optional Hugging Face authentication token for private models.
    :param revision: Optional Git revision (branch, tag, or commit hash) to use when
        downloading the model files from the Hugging Face Hub.
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


@app.command()
def version():
    """Show the speculators version."""
    typer.echo(f"speculators version: {pkg_version('speculators')}")


if __name__ == "__main__":
    app()
