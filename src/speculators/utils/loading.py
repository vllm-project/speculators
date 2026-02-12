from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from loguru import logger
from safetensors import safe_open

# Forward import to avoid circular dependency:
# speculators.models.eagle3.core imports from this file,
# and we need SpeculatorModel type for load_pretrained_weights()
if TYPE_CHECKING:
    from speculators import SpeculatorModel

std_logger = logging.getLogger(__name__)


def load_model_layers(
    layer_names: list[str], model_path: str
) -> dict[str, torch.Tensor]:
    """
    Load one or more named tensors from a HF repo using safetensors shards.
    Supports both exact keys and suffix pattern matching.

    :param layer_names: list of tensor names or suffix patterns to load, e.g.
    ["model.embed_tokens.weight", "lm_head.weight"]
    :param model_path: either a local directory of huggingface model
    containing model.safetensors.index
    :return: dict mapping input names/patterns to loaded tensors
    """
    # download the index file or build weight map for single-file models
    try:
        index_file = _resolve_file(model_path, "model.safetensors.index.json")
        with Path(index_file).open() as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):
        logger.warning(
            "`model.safetensors.index.json` file not found. "
            "Checking for `model.safetensors` instead."
        )
        model_file = _resolve_file(model_path, "model.safetensors")
        # Build virtual weight map for single-file models
        with safe_open(model_file, framework="pt", device="cpu") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")

    # Resolve names: try exact match first, then suffix match
    name_to_key = {}  # Maps input name to actual checkpoint key
    for name in layer_names:
        if name in weight_map:
            name_to_key[name] = name
        else:
            matched = next((k for k in weight_map if k.endswith(name)), None)
            if matched:
                name_to_key[name] = matched
            else:
                logger.error(f"Tensor '{name}' not found in weight_map.")

    # group requested names by shard filename
    shard_to_names: dict[str, list[tuple[str, str]]] = {}
    for name, key in name_to_key.items():
        shard = weight_map[key]
        shard_to_names.setdefault(shard, []).append((name, key))

    if not shard_to_names:
        raise ValueError("None of the requested tensor names were found in the index.")

    # fetch each required shard and extract only the requested tensors
    out: dict[str, Any] = {}
    for shard_file, name_key_pairs in shard_to_names.items():
        shard_path = _resolve_file(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name, key in name_key_pairs:
                out[name] = f.get_tensor(key)
    return out


def _resolve_file(model_path: str, file_name: str) -> Path:
    """
    If model_path is a local directory, return path/<filename> if it exists.
    Otherwise treat model_path as a HF repo_id and download with hf_hub_download.

    :param model_path: local directory or HF repo_id
    :param file_name: filename to look for or download
    :return: local path to the resolved file
    """
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        logger.info("Loading from local directory: {}", model_path)
        p = model_path_obj / file_name
        if not p.exists():
            raise FileNotFoundError(f"Expected local file missing: {p}")
        return p
    # Treat as repo_id on the Hub
    logger.info(f"Loading from huggingface directory: {model_path}: {file_name}")
    return Path(hf_hub_download(repo_id=model_path, filename=file_name))


def load_full_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """
    Load complete state dict from safetensors format (single or sharded).

    Supports both local paths and HuggingFace Hub model IDs.

    Args:
        model_path: Path to local directory OR HuggingFace Hub model ID

    Returns:
        Dictionary mapping parameter names to tensors (on CPU)

    Raises:
        FileNotFoundError: If no safetensors files found
    """
    # Resolve path (local or download from HF)
    resolved_path = _resolve_model_path(model_path)

    # Try loading as single file first
    single_file = resolved_path / "model.safetensors"
    if single_file.exists():
        return _load_single_safetensors(single_file)

    # Try loading as sharded
    index_file = resolved_path / "model.safetensors.index.json"
    if index_file.exists():
        return _load_sharded_safetensors(resolved_path, index_file)

    raise FileNotFoundError(
        f"No safetensors files found in {model_path}. "
        "Expected 'model.safetensors' or 'model.safetensors.index.json'"
    )


def _resolve_model_path(model_path: str) -> Path:
    """Resolve model path from local directory or HF Hub."""
    path = Path(model_path)

    if path.exists():
        return path

    # Try to download from HF Hub
    try:
        std_logger.info(f"Downloading from HuggingFace Hub: {model_path}")
        local_dir = snapshot_download(
            repo_id=model_path,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
        )
        return Path(local_dir)
    except RepositoryNotFoundError as e:
        raise FileNotFoundError(
            f"Model.safetensors or model.safetensors.index.json not found "
            f"at local path '{model_path}' and not found on HuggingFace Hub"
        ) from e


def _load_single_safetensors(file_path: Path) -> dict[str, torch.Tensor]:
    """Load tensors from a single safetensors file."""
    std_logger.info(f"Loading single safetensors file: {file_path}")
    state_dict = {}
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            state_dict[key] = f.get_tensor(key)
    return state_dict


def _load_sharded_safetensors(
    model_dir: Path, index_file: Path
) -> dict[str, torch.Tensor]:
    """Load tensors from sharded safetensors files."""
    std_logger.info(f"Loading sharded safetensors from {model_dir}")

    with index_file.open() as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"index.json contains no weight_map: {index_file}")

    # Collect unique shard files
    shard_files = set(weight_map.values())

    # Load tensors from all shards
    state_dict = {}
    for shard_file in sorted(shard_files):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        std_logger.info(f"  Loading shard: {shard_file}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if weight_map.get(key) == shard_file:
                    state_dict[key] = f.get_tensor(key)

    return state_dict


def extract_vocab_mappings(
    state_dict: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract d2t and t2d vocabulary mappings from EAGLE3 state dict.

    Args:
        state_dict: Model state dictionary
        device: Device to move tensors to

    Returns:
        Tuple of (d2t, t2d) tensors, moved to specified device

    Raises:
        ValueError: If exact match for d2t or t2d not found
    """
    # Find exact matches for d2t and t2d
    d2t_key = _find_exact_key(state_dict, "d2t")
    t2d_key = _find_exact_key(state_dict, "t2d")

    std_logger.info(f"Extracting d2t from: {d2t_key}")
    std_logger.info(f"Extracting t2d from: {t2d_key}")

    # Extract and validate
    d2t = state_dict.pop(d2t_key).to(device)
    t2d = state_dict.pop(t2d_key).to(device)

    _validate_mapping_tensor(d2t, "d2t")
    _validate_mapping_tensor(t2d, "t2d")

    std_logger.info(f"d2t shape: {d2t.shape}, t2d shape: {t2d.shape}")

    return d2t, t2d


def _find_exact_key(state_dict: dict, target: str) -> str:
    """
    Find exact match for key in state dict (case-insensitive).

    Args:
        state_dict: Model state dictionary
        target: Target key name to find

    Returns:
        Matching key from state dict

    Raises:
        ValueError: If exact match not found
    """
    for key in state_dict:
        if key.lower() == target.lower():
            return key

    available_keys = list(state_dict.keys())[:20]
    raise ValueError(
        f"Key '{target}' not found in state dict. "
        f"Available keys sample: {available_keys}..."
    )


def _validate_mapping_tensor(tensor: torch.Tensor, name: str):
    """Validate vocabulary mapping tensor shape."""
    if tensor.dim() != 1:
        raise ValueError(
            f"Unexpected {name} shape: {tensor.shape}. Expected 1D tensor."
        )

def load_pretrained_weights(
    model: SpeculatorModel,
    state_dict: dict[str, torch.Tensor],
    model_path: str,
) -> None:
    """
    Load pretrained weights into model with validation.
    
    Args:
        model: Model to load weights into
        state_dict: State dictionary from pretrained model
        model_path: Path or identifier of the pretrained model (for logging)
    """
    std_logger.info(f"Loading pretrained weights from {model_path}")
    std_logger.info(f"Parameters to load: {len(state_dict)}")

    # Load with strict=False (d2t/t2d passed to constructor)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Build set of legitimately missing keys
    expected_missing = {
        "t2d",
        "d2t",
        "verifier_lm_head.weight",
    }
    
    # Honor model's own ignore list
    model_ignored = getattr(model, "_keys_to_ignore_on_load_missing", [])
    expected_missing.update(model_ignored)

    # Filter problematic missing keys
    problematic = [k for k in missing_keys if k not in expected_missing]

    # Report issues
    if problematic:
        std_logger.warning(f"Unexpected missing keys: {problematic}")
    if unexpected_keys:
        std_logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

    # Summary
    if problematic or unexpected_keys:
        std_logger.warning(
            "Weight loading completed with warnings. May indicate architecture mismatch."
        )
    else:
        std_logger.info("✓ Successfully loaded all weights")
