"""Convert command — convert external models to speculators format."""

import json
from typing import Annotated, Any

import click
import typer

from speculators.convert import convert_model


def convert(
    model: Annotated[
        str,
        typer.Argument(help="Model checkpoint or Hugging Face model ID to convert"),
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
                "The source repo/algorithm to convert from into the matching "
                "algorithm in Speculators"
            ),
            click_type=click.Choice(["eagle", "eagle3", "mtp"]),
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Directory path where converted model will be saved")
    ] = "converted",
    validate_device: Annotated[
        str | None,
        typer.Option(
            help=(
                "Device to validate the model on (e.g. 'cuda:0') "
                "If not provided, validation is skipped."
            ),
        ),
    ] = None,
    algorithm_kwargs: Annotated[
        dict[str, Any] | None,
        typer.Option(
            parser=json.loads,
            help=(
                "Additional keyword args for the conversion alg as a JSON string. "
                'Options for Eagle: {"layernorms": true, "fusion_bias": true}. '
                'Options for Eagle3: {"norm_before_residual": true, '
                '"eagle_aux_hidden_state_layer_ids": [1,23,44]}. '
                'Options for MTP: {"num_speculative_steps": 3}.'
            ),
        ),
    ] = None,
):
    """Convert models from external research repositories or formats
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

    \b
    algorithm=="mtp":
        MTP (Multi-Token Prediction): models with native MTP layers
        (e.g. Qwen3-Next, Qwen3.5, Qwen3.5-MoE)
        ::
        speculators convert "Qwen/Qwen3-Next-80B-A3B-Instruct" \\
            --algorithm mtp \\
            --verifier "Qwen/Qwen3-Next-80B-A3B-Instruct" \\
            --algorithm-kwargs '{"num_speculative_steps": 3}'
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
