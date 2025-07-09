"""
Utility functions for checkpoint conversion operations.
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
    Download a checkpoint from HuggingFace Hub.

    Example:
    ::
        from speculators.utils import download_model_checkpoint_from_hub

        path = download_model_checkpoint_from_hub("yuhuili/EAGLE-LLaMA3.1-Instruct-8B")
        print(path)
        # Output: .../uhuili/EAGLE-LLaMA3.1-Instruct-8B/snapshots/...

    :param model_id: HuggingFace model ID
    :param cache_dir: Optional directory to cache downloads
    :param force_download: Whether to force re-download even if cached
    :param local_files_only: If True, only use local files
    :param token: Optional authentication token for private models
    :param revision: Optional model revision (branch, tag, or commit)
    :param kwargs: Additional arguments for `snapshot_download`
    :return: Local path to the downloaded checkpoint
    :raises FileNotFoundError: If the checkpoint cannot be downloaded
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
    Ensure we have a local copy of the model checkpoint.

    If it is already a model, then return it as-is.
    If the path exists locally, return it.
    Otherwise, treat it as a HuggingFace model ID and download it.

    Example:
    ::
        from speculators.utils import check_download_model_checkpoint

        # Local path - returned as-is
        local = check_download_model_checkpoint("./my_checkpoint")
        # HuggingFace ID - downloaded first
        downloaded = check_download_model_checkpoint(
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        )

    :param model: Local path, HuggingFace model ID, or a PreTrainedModel instance
    :param cache_dir: Optional cache directory for downloads
    :param force_download: Whether to force re-download even if cached
    :param local_files_only: If True, only use local files
    :param token: Optional authentication token for private models
    :param revision: Optional model revision (branch, tag, or commit)
    :param kwargs: Additional arguments for `snapshot_download`
    :return: Path to the local directory containing the model checkpoint
        if model is a path or HuggingFace ID,
        or the model instance if it was passed directly.
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
    Ensure we have a local copy of the model's configuration file.

    If it is already a PretrainedConfig instance, return it as-is.
    If it is a PreTrainedModel instance, return its config.
    If the path exists locally, return it.
    Otherwise, treat it as a HuggingFace model ID, download it,
    and return the PreTrainedConfig object.

    :param config: Local path, HuggingFace model ID,
        PreTrainedModel instance, or PretrainedConfig instance.
    :param cache_dir: Optional directory to cache downloads
    :param force_download: Whether to force re-download even if cached
    :param local_files_only: If True, only use local files
    :param token: Optional authentication token for private models
    :param revision: Optional model revision (branch, tag, or commit)
    :param kwargs: Additional arguments for `AutoConfig.from_pretrained`
    :return: Path to the local config.json file if config is a path or HuggingFace ID,
        or the PretrainedConfig instance if it was passed directly.
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
    Load the configuration for a model from a local checkpoint directory
    or a PreTrainedModel instance.

    Example:
    ::
        from speculators.utils import load_model_config

        config = load_model_config("./checkpoint")
        print(config.model_type)
        # Output: llama

    :param model: The path to the model's local checkpoint directory,
        or a PreTrainedModel instance.
    :param cache_dir: Optional directory to cache downloads
    :param force_download: Whether to force re-download even if cached
    :param local_files_only: If True, only use local files
    :param token: Optional authentication token for private models
    :param revision: Optional model revision (branch, tag, or commit)
    :param kwargs: Additional arguments for `AutoConfig.from_pretrained`
    :return: The PretrainedConfig object for the model.
    :raises FileNotFoundError: If the config.json file cannot be found
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
    Load the configuration dictionary from a model's local checkpoint directory,
    a PreTrained instance, or previously extracted dictionary.
    If config is a dict, it is returned as-is.
    If config is a PretrainedConfig or PreTrainedModel instance,
    its `to_dict()` method is called to extract the configuration.
    If config is a str or Path, it is treated as a path to a local config.json file.

    Example:
    ::
        from speculators.utils import load_model_checkpoint_config_dict

        config = load_model_checkpoint_config_dict("./checkpoint")
        print(config["model_type"])
        # Output: llama

    :param path: The path to the model's local checkpoint directory
        or the path to the local config.json file itself.
    :return: The configuration dictionary loaded from config.json.
    :raises FileNotFoundError: If the config.json file cannot be found
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
    Load all weight files from any index files in a model's local checkpoint directory.
    The index files are expected to be in `.index.json` format, which maps weight names
    to their corresponding file paths.
    If the path is a directory, will look for `.index.json` files within that directory.
    If the path is a single `.index.json` file, it will read that file directly.
    If no index files are found, an empty list is returned.

    Example:
    ::
        from speculators.utils import load_model_checkpoint_index_weight_files

        index_files = load_model_checkpoint_index_weight_files("./checkpoint")
        print(f"Found {len(index_files)} index files")
        # Output: Found 2 index files

    :param path: The path to the model's local checkpoint directory
        or the path to the local index file itself.
    :return: List of Paths to the weight files found in the index files.
        Returns an empty list if no index files are found.
    :raises FileNotFoundError: If the path, any index file, or any weight file
        specified in the index file does not exist.
    :raises ValueError: If any index file does not contain a valid weight_map.
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
    Find and return all weight files given in a model's local checkpoint directory,
    an index.json file, or a single weight file.
    The weight files must be in `.bin` or `.safetensors` format.

    Example:
    ::
        from speculators.utils import load_model_checkpoint_weight_files

        weight_files = load_model_checkpoint_weight_files("./checkpoint")
        print(f"Found {len(weight_files)} weight files")
        # Output: Found 3 weight files

    :param path: The path to the model's local checkpoint directory,
        the path to the local index file, or the path to the local weights file itself.
    :return: List of Paths to the weight files found.
    :raises FileNotFoundError: If the path does not exist or no valid weight files
        are found in the directory or index file.
    :raises ValueError: If the index file does not contain a valid weight_map.
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
    Load the state dictionary of a model from its local checkpoint directory,
    a weights file, or a PreTrainedModel/Module instance.
    If a str or Path is provided, this must be the path to a local
    directory or weights file for the model.

    Example:
    ::
        from speculators.utils import load_model_checkpoint_state_dict

        weights = load_model_checkpoint_state_dict(Path("./checkpoint"))
        print(f"Loaded {len(weights)} weights")
        # Output: Loaded 50 weights

    :param model: The path to the model's local checkpoint directory,
        a weights file, or a PreTrainedModel/Module instance to load
        the state dictionary from.
    :return: Dictionary mapping weight names to tensors.
    """
    if isinstance(model, (PreTrainedModel, nn.Module)):
        logger.debug("Model is already a PreTrainedModel or nn.Module instance")
        return model.state_dict()

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
