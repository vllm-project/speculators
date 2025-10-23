"""
Provides the entry points for converting non-speculators model checkpoints to
Speculators model format with the `convert_model` function.

It supports the following algorithms and conversion from their associated
research repositories:
- EAGLE
- EAGLE2
- EAGLE3
- HASS

Functions:
    convert_model: Converts a model checkpoint to the Speculators format.
"""

from typing import Literal

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.eagle.eagle_converter import EagleConverter

__all__ = ["convert_model"]


def convert_model(
    model: str,
    verifier: str,
    algorithm: Literal["eagle", "eagle3"],
    output_path: str = "converted",
    validate_device: str | None = None,
    **kwargs,
):
    """
    Convert a non speculator's model checkpoint to a speculator's model checkpoint
    for use within the Speculators library, Hugging Face Hub, or vLLM.

    algorithm=="eagle":
        Eagle v1, v2: https://github.com/SafeAILab/EAGLE
        HASS: https://github.com/HArmonizedSS/HASS
        ::
        # general
        convert_model(
            model="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
            verifier="meta-llama/Llama-3.1-8B-Instruct",
            algorithm="eagle",
        )
        # with layernorms and fusion bias enabled
        convert_model(
            model="./eagle/checkpoint",
            verifier="meta-llama/Llama-3.1-8B-Instruct",
            algorithm="eagle",
            layernorms=True,
            fusion_bias=True,
        )

    algorithm=="eagle3":
        Eagle v3: https://github.com/SafeAILab/EAGLE
        ::
        # general
        convert_model(
            model="./eagle/checkpoint",
            verifier="meta-llama/Llama-3.1-8B-Instruct",
            algorithm="eagle3",
        )
        # with normalization before the residual
        convert_model(
            model="./eagle/checkpoint",
            verifier="meta-llama/Llama-3.1-8B-Instruct",
            algorithm="eagle3",
            norm_before_residual=True,
        )

    :param model: Path to the input model checkpoint or Hugging Face model ID.
    :param verifier: Verifier model checkpoint or Hugging Face model ID
        to attach as the verification/base model for speculative decoding
    :param algorithm: The conversion algorithm to use, either "eagle" or "eagle3".
    :param output_path: Directory path where the converted model will be saved.
    :param kwargs: Additional keyword arguments for the conversion algorithm.
        Options for Eagle: {"layernorms": true, "fusion_bias": true}.
        Options for Eagle3: {"norm_before_residual": true,
        "eagle_aux_hidden_state_layer_ids": [1,23,44]}.
    """

    if algorithm == "eagle":
        EagleConverter().convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
    elif algorithm == "eagle3":
        Eagle3Converter().convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
