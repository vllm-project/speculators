"""Convert command — convert external models to speculators format."""

import json
from enum import Enum
from typing import Annotated, Any

import typer

from speculators.convert import convert_model


class AlgorithmChoice(str, Enum):
    eagle3 = "eagle3"
    mtp = "mtp"
    dflash = "dflash"


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
        AlgorithmChoice,
        typer.Option(
            help=(
                "The source repo/algorithm to convert from into the matching "
                "algorithm in Speculators"
            ),
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
                'Options for Eagle3: {"norm_before_residual": true, '
                '"eagle_aux_hidden_state_layer_ids": [1,23,44]}. '
                'Options for MTP: {"num_speculative_steps": 3}. '
                'Options for DFlash: {"aux_hidden_state_layer_ids": [2,10,18,26,34]}.'
            ),
        ),
    ] = None,
):
    """Convert models from external research repositories or formats
    into the standardized Speculators format for use within the Speculators
    framework, Hugging Face model hub compatibility, and deployment with vLLM.
    Supported algorithms, repositories, and examples given below.

    \b
    algorithm=="eagle3":
        Eagle v3: https://github.com/SafeAILab/EAGLE
        ::
        # general
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle3 \\
            --verifier "meta-llama/Llama-3.1-8B-Instruct"
        # with normalization before the residual
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle3 \\
            --algorithm-kwargs '{"norm_before_residual": true}' \\
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

    \b
    algorithm=="dflash":
        DFlash: https://z-lab.ai/projects/dflash/
        ::
        speculators convert "z-lab/Qwen3-8B-DFlash-b16" \\
            --algorithm dflash \\
            --verifier "Qwen/Qwen3-8B"
    """
    if not algorithm_kwargs:
        algorithm_kwargs = {}
    elif not isinstance(algorithm_kwargs, dict):
        raise typer.BadParameter(
            "--algorithm-kwargs must be a JSON object, not "
            + type(algorithm_kwargs).__name__
        )

    convert_model(
        model=model,
        verifier=verifier,
        output_path=output_path,
        validate_device=validate_device,
        algorithm=algorithm.value,
        **algorithm_kwargs,
    )
