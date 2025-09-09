"""
Entry points for converting non-Speculators model checkpoints to Speculators format.

Provides the primary conversion interface through the `convert_model` function, which
supports various input formats including local checkpoints, Hugging Face model IDs,
and PyTorch module instances. Converts models from research implementations (EAGLE,
EAGLE2, HASS) into standardized Speculators format with optional verifier attachment
and validation capabilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

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
    model: str | os.PathLike | PreTrainedModel | nn.Module,
    output_path: str | os.PathLike | None = None,
    config: str | os.PathLike | PreTrainedModel | PretrainedConfig | dict | None = None,
    verifier: str | os.PathLike | PreTrainedModel | None = None,
    validate_device: str | torch.device | int | None = None,
    algorithm: Literal["auto", "eagle", "eagle2", "hass"] = "auto",
    algorithm_kwargs: dict | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    **kwargs,
) -> SpeculatorModel:
    """
    Convert a non-Speculators model checkpoint to Speculators format.

    Supports model instances, local Hugging Face checkpoints, and Hugging Face hub
    model IDs. Optional verifier attachment and validation capabilities are provided
    for enhanced model functionality.

    Example:
    ::
        from speculators.convert import convert_model

        speculator_model = convert_model(
            model="./my_checkpoint",
            output_path="./converted_speculator_model",
            algorithm="eagle",
            verifier="./my_verifier_checkpoint",
        )

    :param model: Path to checkpoint directory, Hugging Face model ID, or
        PreTrainedModel instance to convert
    :param output_path: Optional path to save the converted model
    :param config: Optional config path, model ID, or config instance. If not
        provided, inferred from model checkpoint
    :param verifier: Optional verifier checkpoint path, model ID, or instance to
        attach to the converted model
    :param validate_device: Optional device for post-conversion validation
    :param algorithm: Conversion algorithm - "auto", "eagle", "eagle2", or "hass"
    :param algorithm_kwargs: Optional keyword arguments for the conversion algorithm
    :param cache_dir: Optional directory for caching downloaded model files
    :param force_download: Force re-downloading files even if cached
    :param local_files_only: Use only local files without downloading from hub
    :param token: Optional Hugging Face authentication token for private models
    :param revision: Optional Git revision for downloading from Hugging Face hub
    :param kwargs: Additional keyword arguments for model and config download
    :return: The converted speculator model instance
    :raises ValueError: When config is required but not provided for nn.Module input
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

    converter_class = SpeculatorConverter.resolve_converter(
        algorithm,
        model=model,
        config=config,
        verifier=verifier,
        **algorithm_kwargs,
    )
    logger.info(f"Beginning conversion with Converter: {converter_class}")

    converter = converter_class(
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
