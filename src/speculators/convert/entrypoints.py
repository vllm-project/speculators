"""
A module that provides the entry points for converting non-Speculators model checkpoints
to Speculators format with the `convert_model` function.
It supports various inputs while converting to a set list of supported algorithms:
- EAGLE
- EAGLE2
- HASS
Functions:
    convert_model: Converts a model checkpoint to the Speculators format.
"""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from loguru import logger
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators.convert.converters import SpeculatorConverter
from speculators.model import SpeculatorModel
from speculators.utils import (
    check_download_model_checkpoint,
    check_download_model_config,
)

__all__ = ["convert_model"]


def convert_model(
    model: Union[str, os.PathLike, PreTrainedModel, nn.Module],
    output_path: Optional[Union[str, os.PathLike]] = None,
    config: Optional[
        Union[str, os.PathLike, PreTrainedModel, PretrainedConfig, dict]
    ] = None,
    verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
    validate_device: Optional[Union[str, torch.device, int]] = None,
    algorithm: Literal["auto", "eagle", "eagle2", "hass"] = "auto",
    algorithm_kwargs: Optional[dict] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> SpeculatorModel:
    """
    Convert a non speculator's model checkpoint to a speculator's model
    for use within the Speculators library.
    Supports model instances, local Hugging Face checkpoints, and Hugging Face
    hub model IDs.
    Pass in the `verifier` argument to attach a verifier to the
    converted speculator model. The verifier can be a local path to a
    verifier checkpoint, a Hugging Face model ID, or a PreTrainedModel instance.
    Returns the converted model instance, which is a subclass of
    `speculators.model.SpeculatorModel`.
    If `output_path` is provided, the converted model will be saved
    to that path in the Speculators format.
    Currently supports conversion from EAGLE, EAGLE2, and HASS GitHub research
    repositories into an EagleSpeculator model instance.
    Example:
    ::
        from speculators.convert import convert_model
        # Convert a local checkpoint directory
        speculator_model = convert_model(
            model="./my_checkpoint",
            output_path="./converted_speculator_model",
            algorithm="eagle",
            verifier="./my_verifier_checkpoint",
        )
        print(speculator_model)
    :param model: Path to a local checkpoint directory, Hugging Face model ID,
        or a PreTrainedModel instance to convert.
    :param output_path: Optional path to save the converted speculator model.
        If not provided, the model will not be saved to disk.
    :param config: Optional path to a local config.json file, Hugging Face model ID,
        or a PretrainedConfig instance. If not provided, the model's config will be
        inferred from the model checkpoint.
    :param verifier: Optional path to a verifier checkpoint, Hugging Face model ID,
        or a PreTrainedModel instance. If provided, the verifier will be attached
        to the converted speculator model.
    :param validate_device: Optional device to validate the model on after conversion.
        Can be a string (e.g., "cpu", "cuda"), a torch.device instance, or an integer
        (e.g., 0 for "cuda:0"). If not provided, no validation is performed.
    :param algorithm: The conversion algorithm to use.
        Can be "auto", "eagle", "eagle2", or "hass".
        Defaults to "auto", which will automatically select the appropriate algorithm
        based on the model type and configuration, if possible.
    :param algorithm_kwargs: Optional additional keyword arguments to pass to the
        conversion algorithm.
    :param cache_dir: Optional directory to cache downloaded model files.
        If not provided, the default cache directory will be used.
    :param force_download: If True, forces re-downloading the model files even if they
        already exist in the cache. Defaults to False.
    :param local_files_only: If True, only uses local files and does not attempt to
        download from the Hugging Face hub. Defaults to False.
    :param token: Optional Hugging Face authentication token for private models.
    :param revision: Optional Git revision (branch, tag, or commit hash) to use when
        downloading the model files from the Hugging Face hub.
    :param kwargs: Additional keyword arguments to pass to the model and config
        download functions.
    :return: The converted speculator model instance.
    """
    logger.info(f"Converting model {model} to the Speculators format...")

    model = check_download_model_checkpoint(
        model,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        **kwargs,
    )
    logger.info(f"Resolved the model checkpoint: {model}")

    if not config:
        # Use model as config if not provided
        if isinstance(model, nn.Module):
            raise ValueError(
                "A model config must be provided when converting "
                "a PyTorch nn.Module instance."
            )
        config = model

    config = check_download_model_config(
        config,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        **kwargs,
    )
    logger.info(f"Resolved the model config: {config}")

    if not algorithm_kwargs:
        algorithm_kwargs = {}

    ConverterClass = SpeculatorConverter.resolve_converter(  # noqa: N806
        algorithm,
        model=model,
        config=config,
        verifier=verifier,
        **algorithm_kwargs,
    )
    logger.info(f"Beginning conversion with Converter: {ConverterClass}")

    converter = ConverterClass(
        model=model,
        config=config,
        verifier=verifier,
        **algorithm_kwargs,
    )

    converted = converter(
        output_path=output_path,
        validate_device=validate_device,
    )
    logger.info(f"Conversion complete: {converted}")

    return converted
