"""
Provides the entry points for converting non-speculators model checkpoints to
Speculators model format with the `convert_model` function.

It supports the following algorithms and conversion from their associated
research repositories:
- EAGLE
- EAGLE2
- EAGLE3
- HASS
- MTP
- DFlash

Functions:
    convert_model: Converts a model checkpoint to the Speculators format.
"""

import os
import tempfile
from typing import Literal

from loguru import logger
from transformers import PretrainedConfig

from speculators.convert.dflash.converter import DFlashConverter
from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.eagle.eagle_converter import EagleConverter
from speculators.convert.mtp.converter import MTPConverter

__all__ = ["convert_model", "maybe_convert_external_checkpoint"]


def convert_model(
    model: str,
    verifier: str,
    algorithm: Literal["eagle", "eagle3", "mtp", "dflash"],
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

    algorithm=="mtp":
        MTP (Multi-Token Prediction): models with native MTP layers
        (e.g. Qwen3-Next, Qwen3.5, Qwen3.5-MoE)
        ::
        convert_model(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            verifier="Qwen/Qwen3-Next-80B-A3B-Instruct",
            algorithm="mtp",
            num_speculative_steps=3,
        )

    algorithm=="dflash":
        DFlash: https://z-lab.ai/projects/dflash/
        ::
        convert_model(
            model="z-lab/Qwen3-8B-DFlash-b16",
            verifier="Qwen/Qwen3-8B",
            algorithm="dflash",
        )

    :param model: Path to the input model checkpoint or Hugging Face model ID.
    :param verifier: Verifier model checkpoint or Hugging Face model ID
        to attach as the verification/base model for speculative decoding
    :param algorithm: The conversion algorithm to use:
        "eagle", "eagle3", "mtp", or "dflash".
    :param output_path: Directory path where the converted model will be saved.
    :param kwargs: Additional keyword arguments for the conversion algorithm.
        Options for Eagle: {"layernorms": true, "fusion_bias": true}.
        Options for Eagle3: {"norm_before_residual": true,
        "eagle_aux_hidden_state_layer_ids": [1,23,44]}.
        Options for MTP: {"num_speculative_steps": 3}.
        Options for DFlash: {"aux_hidden_state_layer_ids": [2,10,18,26,34]}.
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
    elif algorithm == "mtp":
        MTPConverter().convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
    elif algorithm == "dflash":
        DFlashConverter().convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def maybe_convert_external_checkpoint(
    model: str | os.PathLike,
    verifier: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    output_path: str | None = None,
    config_dict: dict | None = None,
) -> str:
    """Convert an external (non-speculators) checkpoint to speculators format.

    A speculators checkpoint (config has ``speculators_model_type``) is returned
    unchanged; otherwise the external format is detected and converted (which
    requires ``verifier``) to ``output_path``, defaulting to a temp dir. Powers
    the unified ``from_pretrained`` finetuning pathway. Pass ``config_dict`` to
    reuse an already-loaded config and skip re-reading it.
    """
    if config_dict is None:
        config_dict, _ = PretrainedConfig.get_config_dict(model, cache_dir=cache_dir)
    if "speculators_model_type" in config_dict:
        return str(model)

    architectures = config_dict.get("architectures") or []
    algorithm: Literal["dflash"] = "dflash"
    if not (
        "dflash_config" in config_dict or any("DFlash" in a for a in architectures)
    ):
        raise NotImplementedError(
            f"Cannot auto-convert checkpoint '{model}': unrecognized external "
            "format. Supported auto-conversion: DFlash."
        )

    if verifier is None:
        raise ValueError(
            f"Converting an external {algorithm} checkpoint requires a verifier. "
            "Pass `verifier=<model id or path>`."
        )

    output_path = output_path or tempfile.mkdtemp(prefix="speculators_converted_")
    logger.info(f"Auto-converting external {algorithm} checkpoint to {output_path}")
    convert_model(
        model=str(model),
        verifier=verifier,
        algorithm=algorithm,
        output_path=output_path,
        cache_dir=cache_dir,
    )
    return output_path
