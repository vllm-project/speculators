"""
Utility functions for interacting with Hugging Face's Transformers library.

This module provides utilities for downloading, loading, and managing model checkpoints
and configurations from Hugging Face Hub and local directories. It handles various
model formats including PyTorch bins, SafeTensors, and indexed weight files commonly
used in transformer models.

The utilities support both local file operations and remote downloads from Hugging Face
Hub, with automatic caching and format detection. All functions are designed to work
seamlessly with the transformers library ecosystem while providing additional
convenience features for model management.

Functions:
    download_model_checkpoint_from_hub: Download checkpoints from Hugging Face Hub
    check_download_model_checkpoint: Ensure local availability of model checkpoints
    check_download_model_config: Ensure local availability of model configurations
    load_model_config: Load PretrainedConfig from various sources
    load_model_checkpoint_config_dict: Load configuration as dictionary
    load_model_checkpoint_index_weight_files: Load weight files from index files
    load_model_checkpoint_weight_files: Find and load model weight files
    load_model_checkpoint_state_dict: Load complete model state dictionary

Usage:
::
    from speculators.utils import transformer_utils

    # Download and load a model checkpoint
    checkpoint_path = transformer_utils.download_model_checkpoint_from_hub(
        "huggingface/model-id"
    )

    # Load model configuration
    config = transformer_utils.load_model_config(checkpoint_path)

    # Load model weights
    state_dict = transformer_utils.load_model_checkpoint_state_dict(checkpoint_path)
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

__all__ = [
    "check_download_model_checkpoint",
    "check_download_model_config",
    "download_model_checkpoint_from_hub",
    "load_model_checkpoint_config_dict",
    "load_model_checkpoint_index_weight_files",
    "load_model_checkpoint_state_dict",
    "load_model_checkpoint_weight_files",
    "load_model_config",
]


def download_model_checkpoint_from_hub(
    model_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> Path:
    """
    Download a model checkpoint from Hugging Face Hub.

    Downloads model files including configuration, weights, and index files
    to a local cache directory. Supports authentication for private models
    and various download options.

    :param model_id: Hugging Face model identifier
    :param cache_dir: Directory to cache downloaded files
    :param force_download: Whether to force re-download existing files
    :param local_files_only: Only use cached files without downloading
    :param token: Authentication token for private models
    :param revision: Model revision (branch, tag, or commit hash)
    :param kwargs: Additional arguments for `snapshot_download`
    :return: Path to the downloaded checkpoint directory
    :raises FileNotFoundError: If the model cannot be downloaded
    """
    logger.info(f"Downloading a model checkpoint from HuggingFace: {model_id}")
    try:
        if "allow_patterns" not in kwargs:
            kwargs["allow_patterns"] = [
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.index.json",
            ]
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        logger.info(f"Downloaded a model checkpoint from HuggingFace to: {local_path}")
        return Path(local_path)
    except Exception as hf_exception:
        logger.error(f"Failed to download checkpoint: {hf_exception}")
        raise FileNotFoundError(f"Checkpoint not found: {model_id}") from hf_exception


def check_download_model_checkpoint(
    model: Union[str, os.PathLike, PreTrainedModel, nn.Module],
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> Union[Path, PreTrainedModel, nn.Module]:
    """
    Ensure local availability of a model checkpoint.

    Returns the model directly if it's already a model instance, returns the path
    if it exists locally, or downloads from Hugging Face Hub if needed
    and returns the local path after download.

    :param model: Local path, Hugging Face model ID, or model instance
    :param cache_dir: Directory to cache downloaded files
    :param force_download: Whether to force re-download existing files
    :param local_files_only: Only use cached files without downloading
    :param token: Authentication token for private models
    :param revision: Model revision (branch, tag, or commit hash)
    :param kwargs: Additional arguments for `snapshot_download`
    :return: Path to local checkpoint directory or the model instance
    :raises TypeError: If model is not a supported type
    :raises ValueError: If local path is not a directory
    """
    if isinstance(model, (PreTrainedModel, nn.Module)):
        logger.debug("Model is already a PreTrainedModel or nn.Module instance")
        return model

    if not isinstance(model, (str, os.PathLike)):
        raise TypeError(
            f"Expected model to be a string or Path, got {type(model)} for {model}"
        )

    checkpoint_path = Path(model)

    if not checkpoint_path.exists():
        logger.debug(
            f"Model path does not exist, downloading from hub: {checkpoint_path}"
        )
        return download_model_checkpoint_from_hub(
            model_id=str(checkpoint_path),
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )

    if not checkpoint_path.is_dir():
        raise ValueError(
            f"Expected a directory for checkpoint, got file: {checkpoint_path}"
        )

    return checkpoint_path.resolve()


def check_download_model_config(
    config: Union[str, os.PathLike, PreTrainedModel, PretrainedConfig, dict],
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> Union[Path, PretrainedConfig, dict]:
    """
    Ensure local availability of a model configuration.

    Returns the configuration directly if it's already a config instance or dict,
    extracts config from model instances, returns local path if it exists,
    or downloads from Hugging Face Hub if needed and returns the local path
    after download.

    :param config: Local path, Hugging Face model ID, model instance, or config
    :param cache_dir: Directory to cache downloaded files
    :param force_download: Whether to force re-download existing files
    :param local_files_only: Only use cached files without downloading
    :param token: Authentication token for private models
    :param revision: Model revision (branch, tag, or commit hash)
    :param kwargs: Additional arguments for `AutoConfig.from_pretrained`
    :return: Path to local config.json file or the config instance
    :raises TypeError: If config is not a supported type
    :raises FileNotFoundError: If config.json cannot be found
    """
    if isinstance(config, PretrainedConfig):
        logger.debug("Config is already a PretrainedConfig instance")
        return config

    if isinstance(config, PreTrainedModel):
        logger.debug("Config is a PreTrainedModel instance, returning its config")
        return config.config  # type: ignore[attr-defined]

    if isinstance(config, dict):
        logger.debug("Config is a dictionary, returning as is")
        return config

    if not isinstance(config, (str, os.PathLike)):
        raise TypeError(
            f"Expected config to be a string, Path, or PreTrainedModel, "
            f"got {type(config)} for {config}"
        )

    config_path = Path(config)
    if not config_path.exists():
        logger.debug(f"Config path does not exist, downloading from hub: {config_path}")
        return AutoConfig.from_pretrained(
            str(config_path),
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )

    logger.debug(f"Using local config path: {config_path}")

    if not config_path.is_file():
        config_path = config_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")

    return config_path.resolve()


def load_model_config(
    model: Union[str, os.PathLike, PreTrainedModel, PretrainedConfig],
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> PretrainedConfig:
    """
    Load a PretrainedConfig from various sources.

    Supports loading from local checkpoint directories, Hugging Face model IDs,
    or extracting from existing model instances. Always returns a PretrainedConfig
    object regardless of input type.

    :param model: Local path, Hugging Face model ID, or model instance
    :param cache_dir: Directory to cache downloaded files
    :param force_download: Whether to force re-download existing files
    :param local_files_only: Only use cached files without downloading
    :param token: Authentication token for private models
    :param revision: Model revision (branch, tag, or commit hash)
    :param kwargs: Additional arguments for `AutoConfig.from_pretrained`
    :return: PretrainedConfig object for the model
    :raises TypeError: If model is not a supported type
    :raises FileNotFoundError: If the configuration cannot be found
    """
    logger.debug(f"Loading model config from: {model}")

    if isinstance(model, PretrainedConfig):
        logger.debug("Model is already a PretrainedConfig instance")
        return model

    if isinstance(model, PreTrainedModel):
        logger.debug("Model is a PreTrainedModel instance, returning its config")
        return model.config  # type: ignore[attr-defined]

    if not isinstance(model, (str, os.PathLike)):
        raise TypeError(
            "Expected model to be a string, Path, or PreTrainedModel, "
            f"got {type(model)}"
        )

    try:
        logger.debug(f"Loading config with AutoConfig from: {model}")
        return AutoConfig.from_pretrained(
            model,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
    except ValueError as err:
        logger.error(f"Failed to load config from {model}: {err}")
        raise FileNotFoundError(f"Config not found for model: {model}") from err


def load_model_checkpoint_config_dict(
    config: Union[str, os.PathLike, PretrainedConfig, PreTrainedModel, dict],
) -> dict:
    """
    Load model configuration as a dictionary from various sources.

    Supports loading from local config.json files, checkpoint directories,
    or extracting from existing model/config instances. Always returns
    a dictionary representation of the configuration.

    :param config: Local path, PretrainedConfig, PreTrainedModel, or dict
    :return: Configuration dictionary
    :raises TypeError: If config is not a supported type
    :raises FileNotFoundError: If config.json cannot be found
    """
    if isinstance(config, dict):
        logger.debug("Config is already a dictionary, returning as is")
        return config

    if isinstance(config, PreTrainedModel):
        logger.debug("Config is a PreTrainedModel instance, returning its config dict")
        return config.config.to_dict()  # type: ignore[attr-defined]

    if isinstance(config, PretrainedConfig):
        logger.debug("Config is a PretrainedConfig instance, returning its dict")
        return config.to_dict()

    if not isinstance(config, (str, os.PathLike)):
        raise TypeError(
            f"Expected config to be a string, Path, PreTrainedModel, "
            f"or PretrainedConfig, got {type(config)}"
        )

    path = Path(config)

    if path.is_dir():
        path = path / "config.json"

    if not path.exists():
        raise FileNotFoundError(f"No config.json found at {path}")

    logger.debug(f"Loading config from: {path}")
    with path.open() as file:
        return json.load(file)


def load_model_checkpoint_index_weight_files(
    path: Union[str, os.PathLike],
) -> list[Path]:
    """
    Load weight files referenced in model index files.

    Searches for .index.json files in the given directory or processes a single
    index file, then returns all weight files referenced in the index mappings.
    Returns an empty list if no index files are found.

    :param path: Local checkpoint directory or path to index file
    :return: List of paths to weight files found in index mappings
    :raises TypeError: If path is not a string or Path-like object
    :raises FileNotFoundError: If path or referenced weight files don't exist
    :raises ValueError: If index file lacks valid weight_map
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Expected path to be a string or Path, got {type(path)}")

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint path does not exist: {path}")

    if path.is_file() and path.suffix == ".index.json":
        logger.debug(f"Single index file provided: {path}")
        index_files = [path]
    elif path.is_dir() and (glob_files := list(path.glob("*.index.json"))):
        logger.debug(f"Found index files in directory: {path}: {glob_files}")
        index_files = glob_files
    else:
        logger.debug(f"No index files found in directory: {path}")
        return []

    files = []

    for index_file in index_files:
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index file under {path} at {index_file} does not exist"
            )
        logger.debug(f"Reading index file: {index_file}")
        with index_file.open() as file_handle:
            index_data = json.load(file_handle)
        if not index_data.get("weight_map"):
            raise ValueError(f"Index file {index_file} does not contain a weight_map")
        for weight_file in set(index_data["weight_map"].values()):
            # Resolve relative paths to the index file's directory
            weight_file_path = Path(index_file).parent / weight_file
            if not weight_file_path.exists():
                raise FileNotFoundError(
                    f"Weight file for {path} at {weight_file_path} does not exist"
                )
            files.append(weight_file_path)

    return files


def load_model_checkpoint_weight_files(path: Union[str, os.PathLike]) -> list[Path]:
    """
    Find and return all weight files for a model checkpoint.

    Searches for weight files in various formats (.bin, .safetensors) either
    directly in a directory, through index files, or as a single weight file.
    Automatically detects and handles different weight file organization patterns.

    :param path: Local checkpoint directory, index file, or weight file path
    :return: List of paths to weight files
    :raises TypeError: If path is not a string or Path-like object
    :raises FileNotFoundError: If path doesn't exist or no weight files found
    :raises ValueError: If index file lacks valid weight_map
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Expected path to be a string or Path, got {type(path)}")

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint path does not exist: {path}")

    if index_files := load_model_checkpoint_index_weight_files(path):
        logger.debug(f"Found index files at {path}: {index_files}")
        return index_files

    if path.is_file() and path.suffix in {".bin", ".safetensors"}:
        logger.debug(f"Single weight file provided: {path}")
        return [path]

    if path.is_dir() and (safetensors_files := list(path.glob("*.safetensors"))):
        logger.debug(f"Found safetensors files in dir: {path}: {safetensors_files}")
        return safetensors_files

    if path.is_dir() and (bin_files := list(path.glob("*.bin"))):
        logger.debug(f"Found bin files in dir: {path}: {bin_files}")
        return bin_files

    raise FileNotFoundError(
        f"No valid weight files found in directory: {path}. "
        "Expected .bin, .safetensors, or .index.json files."
    )


def load_model_checkpoint_state_dict(
    model: Union[str, os.PathLike, PreTrainedModel, nn.Module],
) -> dict[str, Tensor]:
    """
    Load complete model state dictionary from various sources.

    Supports loading from model instances, local checkpoint directories,
    individual weight files, or indexed weight collections. Handles both
    PyTorch .bin and SafeTensors .safetensors formats automatically.

    :param model: Model instance, checkpoint directory, or weight file path
    :return: Dictionary mapping parameter names to tensors
    :raises ValueError: If unsupported file format is encountered
    """
    if isinstance(model, (PreTrainedModel, nn.Module)):
        logger.debug("Model is already a PreTrainedModel or nn.Module instance")
        return model.state_dict()  # type: ignore[union-attr]

    logger.debug(f"Loading model weights from: {model}")
    weight_files = load_model_checkpoint_weight_files(model)

    state_dict = {}

    for file in weight_files:
        if file.suffix == ".safetensors":
            logger.debug(f"Loading safetensors weights from: {file}")
            with safe_open(file, framework="pt", device="cpu") as safetensors_file:
                for key in safetensors_file.keys():  # noqa: SIM118
                    state_dict[key] = safetensors_file.get_tensor(key)
        elif file.suffix == ".bin":
            logger.debug(f"Loading PyTorch weights from: {file}")
            loaded_weights = torch.load(file, map_location="cpu")
            for key, value in loaded_weights.items():
                state_dict[key] = value
        else:
            raise ValueError(
                f"Unsupported file type {file.suffix} in {file}. "
                "Expected .safetensors or .bin files."
            )

    return state_dict
