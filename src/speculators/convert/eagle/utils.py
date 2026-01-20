"""
Utility functions for checkpoint conversion operations.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open


def find_vocab_size(config_dict: dict) -> int | None:
    """
    Recursively search for vocab_size in nested config dictionary.

    :param config_dict: Configuration dictionary to search
    :return: vocab_size if found, None otherwise
    """
    if isinstance(config_dict, dict):
        if "vocab_size" in config_dict:
            return config_dict["vocab_size"]
        for value in config_dict.values():
            if isinstance(value, dict):
                result = find_vocab_size(value)
                if result is not None:
                    return result
    return None


def download_checkpoint_from_hub(model_id: str, cache_dir: str | None = None) -> Path:
    """
    Download a checkpoint from HuggingFace Hub.

    :param model_id: HuggingFace model ID
    :param cache_dir: Optional directory to cache downloads
    :return: Local path to the downloaded checkpoint
    :raises FileNotFoundError: If the checkpoint cannot be downloaded

    :Example:

        >>> path = download_checkpoint_from_hub("yuhuili/EAGLE-LLaMA3.1-Instruct-8B")
        >>> print(path)
        /home/user/.cache/huggingface/hub/models--yuhuili--EAGLE-LLaMA3.1-Instruct-8B/snapshots/...
    """
    logger.info(f"Downloading checkpoint from HuggingFace: {model_id}")
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
            cache_dir=cache_dir,
        )
        logger.debug(f"Downloaded to: {local_path}")
        return Path(local_path)
    except Exception as hf_exception:
        logger.error(f"Failed to download checkpoint: {hf_exception}")
        raise FileNotFoundError(f"Checkpoint not found: {model_id}") from hf_exception


def ensure_checkpoint_is_local(
    checkpoint_path: str | Path, cache_dir: str | Path | None = None
) -> Path:
    """
    Ensure we have a local copy of the checkpoint.

    If the path exists locally, return it. Otherwise, treat it as a
    HuggingFace model ID and download it.

    :param checkpoint_path: Local path or HuggingFace model ID
    :param cache_dir: Optional cache directory for downloads
    :return: Path to local checkpoint directory

    :Example:

        >>> # Local path - returned as-is
        >>> local = ensure_checkpoint_is_local("./my_checkpoint")

        >>> # HuggingFace ID - downloaded first
        >>> downloaded = ensure_checkpoint_is_local(
        ...     "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        ... )
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        logger.debug(f"Using local checkpoint: {checkpoint_path}")
        return checkpoint_path

    return download_checkpoint_from_hub(
        model_id=str(checkpoint_path), cache_dir=str(cache_dir) if cache_dir else None
    )


def load_checkpoint_config(checkpoint_dir: Path) -> dict:
    """
    Load the config.json from a checkpoint directory.

    :param checkpoint_dir: Path to checkpoint directory
    :return: Config dictionary
    :raises FileNotFoundError: If config.json is not found

    :Example:

        >>> config = load_checkpoint_config(Path("./checkpoint"))
        >>> print(config["model_type"])
        llama
    """
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {checkpoint_dir}")

    logger.debug(f"Loading config from: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def load_checkpoint_weights(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """
    Load model weights from a checkpoint directory.

    Supports both safetensors and PyTorch bin formats.

    :param checkpoint_dir: Path to checkpoint directory
    :return: Dictionary mapping weight names to tensors
    :raises FileNotFoundError: If no weights are found
    :raises NotImplementedError: If checkpoint is sharded

    :Example:

        >>> weights = load_checkpoint_weights(Path("./checkpoint"))
        >>> print(f"Loaded {len(weights)} weights")
        Loaded 50 weights
    """
    weights = {}

    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        logger.debug(f"Loading safetensors weights from: {safetensors_path}")
        with safe_open(safetensors_path, framework="pt") as f:
            # safetensors requires iterating over keys() method
            for key in f.keys():  # noqa: SIM118
                weights[key] = f.get_tensor(key)
        return weights

    pytorch_path = checkpoint_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        logger.debug(f"Loading PyTorch weights from: {pytorch_path}")
        return torch.load(pytorch_path, map_location="cpu")

    index_paths = [
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "pytorch_model.bin.index.json",
    ]
    for index_path in index_paths:
        if index_path.exists():
            raise NotImplementedError(
                f"Sharded checkpoint detected: {index_path}. "
                "Please use a single-file checkpoint."
            )

    raise FileNotFoundError(f"No weights found at {checkpoint_dir}")


def detect_fusion_bias_and_layernorms(
    weights: dict[str, torch.Tensor],
) -> tuple[bool, bool]:
    """
    Auto-detect fusion bias and extra layernorms presence based on weight names.

    :param weights: Dictionary of weight tensors
    :return: Tuple of (has_fusion_bias, has_layernorms)

    :Example:

        >>> weights = {
        ...     "fc.bias": torch.randn(4096),
        ...     "embed_layernorm.weight": torch.randn(4096)
        ... }
        >>> has_bias, has_ln = detect_fusion_bias_and_layernorms(weights)
        >>> print(f"Fusion bias: {has_bias}, Layernorms: {has_ln}")
        Fusion bias: True, Layernorms: True
    """
    has_fusion_bias = "fc.bias" in weights
    has_layernorms = any(
        name in weights
        for name in ["embed_layernorm.weight", "post_embedding_layernorm.weight"]
    )

    if has_fusion_bias:
        logger.info("Detected fusion bias in checkpoint")
    if has_layernorms:
        logger.info("Detected extra layernorms in checkpoint")

    return has_fusion_bias, has_layernorms
