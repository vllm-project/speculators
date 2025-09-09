"""
Utilities for downloading, loading, and managing Hugging Face transformer models.

This module provides comprehensive utilities for interacting with Hugging Face's
Transformers library, supporting both local file operations and remote downloads
from Hugging Face Hub. It handles various model formats including PyTorch .bin files,
SafeTensors .safetensors files, and indexed weight collections commonly used in
large transformer models. All functions feature automatic caching, format detection,
and seamless integration with the transformers library ecosystem.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: bool | str | None = None,
    revision: str | None = None,
    **kwargs,
) -> Path:
    """
    Download model checkpoint from Hugging Face Hub.

    Downloads model files including configuration, weights, and index files
    to local cache with automatic format detection and authentication support.

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
        raise FileNotFoundError(
            f"Failed to download checkpoint for {model_id}: {hf_exception}"
        ) from hf_exception


def check_download_model_checkpoint(
    model: str | os.PathLike | PreTrainedModel | nn.Module,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    **kwargs,
) -> Path | PreTrainedModel | nn.Module:
    """
    Ensure local availability of model checkpoint.

    Returns model instance directly if provided, otherwise ensures checkpoint
    exists locally by downloading from Hugging Face Hub if necessary.

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
    logger.debug(f"Checking download model checkpoint for: {model}")

    if isinstance(model, (PreTrainedModel, nn.Module)):
        logger.debug("Model is already a PreTrainedModel or nn.Module instance")
        return model

    if not isinstance(model, (str, os.PathLike)):
        raise TypeError(
            f"Expected model to be a string or Path, got {type(model)} for {model}"
        )

    checkpoint_path = Path(model)

    if checkpoint_path.exists():
        logger.debug(f"Model path exists locally: {checkpoint_path}")
        return checkpoint_path.resolve()

    logger.debug(f"Model path does not exist, downloading from hub: {checkpoint_path}")
    return download_model_checkpoint_from_hub(
        model_id=str(checkpoint_path),
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        **kwargs,
    )


def check_download_model_config(
    config: str | os.PathLike | PreTrainedModel | PretrainedConfig | dict,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    **kwargs,
) -> Path | PretrainedConfig | dict:
    """
    Ensure local availability of model configuration.

    Returns configuration instance directly if provided, extracts from model instances,
    or ensures config exists locally by downloading from Hugging Face Hub if necessary.

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
    if config_path.exists() and config_path.is_dir():
        logger.debug("Config path is a directory, looking for config.json in it")
        config_path = config_path / "config.json"

    if config_path.exists() and config_path.is_file():
        logger.debug(f"Using local config file: {config_path}")
        return config_path.resolve()

    try:
        local_path = snapshot_download(
            str(config),
            allow_patterns=["config.json"],
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        logger.info(f"Downloaded a model config from HuggingFace to: {local_path}")

        return Path(local_path) / "config.json"
    except Exception as hf_exception:
        raise FileNotFoundError(
            f"Failed to download config for {config}: {hf_exception}"
        ) from hf_exception


def load_model_config(
    model: str | os.PathLike | PreTrainedModel | PretrainedConfig | dict,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    **kwargs,
) -> PretrainedConfig:
    """
    Load PretrainedConfig from various sources.

    Supports loading from local checkpoint directories, Hugging Face model IDs,
    or extracting from existing model instances.

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

    config = check_download_model_config(
        model,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        **kwargs,
    )

    if isinstance(config, PretrainedConfig):
        logger.debug("Model is already a PretrainedConfig instance")
        return config

    if isinstance(config, dict):
        logger.debug("Model is a dictionary, loading config from dict")
        return PretrainedConfig.from_dict(config)

    try:
        logger.debug(f"Loading config with AutoConfig from: {model}")
        # use model to ensure proper handling of HF args
        # it will resolve to the previously downloaded config path
        return AutoConfig.from_pretrained(
            str(model),
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
    except Exception as hf_exception:
        raise FileNotFoundError(
            f"Failed to download model config for {model}: {hf_exception}"
        ) from hf_exception


def load_model_checkpoint_config_dict(
    model: str | os.PathLike | PretrainedConfig | PreTrainedModel | dict,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    **kwargs,
) -> dict:
    """
    Load model configuration as dictionary from various sources.

    Supports loading from local config.json files, checkpoint directories,
    or extracting from existing model/config instances.

    :param model: Local path, PretrainedConfig, PreTrainedModel, or dict
    :param cache_dir: Directory to cache downloaded files
    :param force_download: Whether to force re-download existing files
    :param local_files_only: Only use cached files without downloading
    :param token: Authentication token for private models
    :param revision: Model revision (branch, tag, or commit hash)
    :param kwargs: Additional arguments for `check_download_model_config`
    :return: Configuration dictionary
    :raises TypeError: If config is not a supported type
    :raises FileNotFoundError: If config.json cannot be found
    """
    logger.debug(f"Loading model config dict from: {model}")
    config = check_download_model_config(
        model,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        **kwargs,
    )

    if isinstance(config, dict):
        logger.debug("Config is already a dictionary, returning as is")
        return config

    if isinstance(config, PretrainedConfig):
        logger.debug("Config is a PretrainedConfig instance, returning its dict")
        return config.to_dict()

    if not isinstance(config, Path):
        raise TypeError(f"Expected config to be a Path, got {type(config)}")

    logger.debug(f"Loading config from: {config}")
    with config.open() as file:
        return json.load(file)


def load_model_checkpoint_index_weight_files(
    path: str | os.PathLike,
) -> list[Path]:
    """
    Load weight files referenced in model index files.

    Searches for .index.json files and returns all weight files referenced
    in their mappings. Returns empty list if no index files found.

    :param path: Local checkpoint directory or path to index file
    :return: List of paths to weight files found in index mappings
    :raises TypeError: If path is not a string or Path-like object
    :raises FileNotFoundError: If path or referenced weight files don't exist
    :raises ValueError: If index file lacks valid weight_map
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Expected path to be a string or Path, got {type(path)}")

    if not (path := Path(path)).exists():
        raise FileNotFoundError(f"Model checkpoint path does not exist: {path}")

    if path.is_file() and path.name.endswith(".index.json"):
        logger.debug(f"Single index file provided: {path}")
        index_files = [path]
    elif path.is_dir() and (glob_files := list(path.glob("*.index.json"))):
        logger.debug(f"Found index files in directory: {path}: {glob_files}")
        index_files = glob_files
    else:
        logger.debug(f"No index files found in directory: {path}")
        return []

    files = set()

    for index_file in index_files:
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index file under {path} at {index_file} does not exist"
            )

        logger.debug(f"Reading index file: {index_file}")
        with index_file.open() as file_handle:
            index_data = json.load(file_handle)

        if (
            not isinstance(index_data, dict)
            or not index_data.get("weight_map")
            or not isinstance(index_data["weight_map"], dict)
        ):
            raise ValueError(
                f"Index file {index_file} does not contain a valid weight_map"
            )

        for weight_file in set(index_data["weight_map"].values()):
            # Resolve relative paths to the index file's directory
            if not (
                weight_file_path := Path(index_file).parent / str(weight_file)
            ).exists():
                raise FileNotFoundError(
                    f"Weight file for {path} at {weight_file_path} does not exist"
                )
            files.add(weight_file_path.resolve())

    return list(files)


def load_model_checkpoint_weight_files(path: str | os.PathLike) -> list[Path]:
    """
    Find and return all weight files for model checkpoint.

    Searches for weight files in various formats (.bin, .safetensors) through
    automatic detection of different organization patterns.

    :param path: HF ID, local checkpoint directory, index file, or weight file path
    :return: List of paths to weight files
    :raises TypeError: If path is not a string or Path-like object
    :raises FileNotFoundError: If path doesn't exist or no weight files found
    :raises ValueError: If index file lacks valid weight_map
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Expected path to be a string or Path, got {type(path)}")

    if not (path := Path(path)).exists():
        raise FileNotFoundError(f"Model checkpoint path does not exist: {path}")

    if weight_index_files := load_model_checkpoint_index_weight_files(path):
        logger.debug(f"Found index files at {path}: {weight_index_files}")
        return weight_index_files

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
    model: str | os.PathLike | PreTrainedModel | nn.Module,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    **kwargs,
) -> dict[str, Tensor]:
    """
    Load complete model state dictionary from various sources.

    Supports loading from model instances, local checkpoint directories,
    or individual weight files with automatic format detection.

    :param model: Model instance, HF ID, checkpoint directory, or weight file path
    :param cache_dir: Directory to cache downloaded files
    :param force_download: Whether to force re-download existing files
    :param local_files_only: Only use cached files without downloading
    :param token: Authentication token for private models
    :param revision: Model revision (branch, tag, or commit hash)
    :param kwargs: Additional arguments for `check_download_model_checkpoint`
    :return: Dictionary mapping parameter names to tensors
    :raises ValueError: If unsupported file format is encountered
    """
    logger.debug(f"Loading model state dict from: {model}")

    model = check_download_model_checkpoint(
        model,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        **kwargs,
    )

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
