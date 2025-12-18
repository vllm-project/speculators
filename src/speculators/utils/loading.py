import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger
from safetensors import safe_open


def load_model_layers(
    layer_names: list[str], model_path: str
) -> dict[str, torch.Tensor]:
    """
    Load one or more named tensors from a HF repo using safetensors shards.
    Supports both exact keys and suffix pattern matching.

    :param layer_names: list of tensor names to load, e.g.
    ["model.embed_tokens.weight", "lm_head.weight"] or suffix patterns like ["embed_tokens.weight"]
    :param model_path: either a local directory containing model.safetensors.index
    :return: dict mapping input names/patterns to loaded tensors
    """
    # download the index file
    try:
        index_file = _resolve_file(model_path, "model.safetensors.index.json")
    except (FileNotFoundError, EntryNotFoundError):
        logger.warning(
            "`model.safetensors.index.json` file not found. "
            "Checking for `model.safetensors` instead."
        )
        _model_file = _resolve_file(model_path, "model.safetensors")
        shard_to_names: dict[str, list[str]] = {"model.safetensors": layer_names}
    else:
        with Path(index_file).open() as f:
            index = json.load(f)

        weight_map: dict[str, str] = index["weight_map"]

        # Resolve names: try exact match first, then suffix match
        name_to_key = {}  # Maps input name to actual checkpoint key
        for name in layer_names:
            if name in weight_map:
                name_to_key[name] = name  # Exact match
            else:
                # Try suffix match
                matched = next((k for k in weight_map if k.endswith(name)), None)
                if matched:
                    name_to_key[name] = matched
                else:
                    logger.error(f"Tensor '{name}' not found in index weight_map.")

        # group requested names by shard filename
        shard_to_names = {}
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
