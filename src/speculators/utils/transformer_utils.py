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
from torch import Tensor
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

__all__ = [
    "check_download_model_checkpoint",
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
    model: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    **kwargs,
) -> Path:
    """
    Ensure we have a local copy of the model checkpoint.

    If the path exists locally, return it. Otherwise, treat it as a
    HuggingFace model ID and download it.

    Example:
    ::
        from speculators.utils import check_download_model_checkpoint

        # Local path - returned as-is
        local = check_download_model_checkpoint("./my_checkpoint")
        # HuggingFace ID - downloaded first
        downloaded = check_download_model_checkpoint(
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        )

    :param model: Local path or HuggingFace model ID
    :param cache_dir: Optional cache directory for downloads
    :param force_download: Whether to force re-download even if cached
    :param local_files_only: If True, only use local files
    :param token: Optional authentication token for private models
    :param revision: Optional model revision (branch, tag, or commit)
    :param kwargs: Additional arguments for `snapshot_download`
    :return: Path to the local directory containing the model checkpoint
    """
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


def load_model_config(
    model: Union[str, os.PathLike, PreTrainedModel, PretrainedConfig],
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
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
        return model.config

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


def load_model_checkpoint_config_dict(path: Union[str, os.PathLike]) -> dict:
    """
    Load the config.json from a model's local checkpoint directory
    into a dictionary.

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
    path = Path(path)

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
    path: Union[str, os.PathLike], keys_only: bool = False
) -> dict[str, Tensor]:
    """
    Load model weights from a local checkpoint directory or weights file.
    The weights file can be a single `.bin` file, a single `.safetensors` file,
    or an index.json file for sharded checkpoints.
    If the path is a directory, it will look for `.bin` or `.safetensors` files
    within that directory. If both are present, `.safetensors` will be preferred.

    Example:
    ::
        from speculators.utils import load_model_checkpoint_weights

        weights = load_model_checkpoint_weights(Path("./checkpoint"))
        print(f"Loaded {len(weights)} weights")
        # Output: Loaded 50 weights

    :param path: The path to the model's local checkpoint directory
        or the path to the local weights file itself.
    :param keys_only: If True, only return the keys mapped to empty tensors
        to avoid loading the large weights into memory if they are not needed.
    :return: Dictionary mapping weight names to tensors.
    """
    logger.debug(f"Loading model weights from: {path}")

    weight_files = load_model_checkpoint_weight_files(path)

    state_dict = {}

    for file in weight_files:
        if file.suffix == ".safetensors":
            logger.debug(f"Loading safetensors weights from: {file}")
            with safe_open(file, framework="pt", device="cpu") as safetensors_file:
                for key in safetensors_file.keys():  # noqa: SIM118
                    state_dict[key] = (
                        safetensors_file.get_tensor(key)
                        if not keys_only
                        else torch.empty(0)
                    )
        elif file.suffix == ".bin":
            logger.debug(f"Loading PyTorch weights from: {file}")
            loaded_weights = torch.load(file, map_location="cpu")
            for key, value in loaded_weights.items():
                state_dict[key] = value if not keys_only else torch.empty(0)
        else:
            raise ValueError(
                f"Unsupported file type {file.suffix} in {file}. "
                "Expected .safetensors or .bin files."
            )

    return state_dict
