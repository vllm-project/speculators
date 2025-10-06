from typing import Any, List, Dict, Union
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from pathlib import Path
from loguru import logger
import json
import torch
import os

def load_model_layers(layer_names: list[str], model_path: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load one or more named tensors from a HF repo using safetensors shards.
    Returns a single tensor if len(layer_names)==1, else a dict[name] -> tensor.

    :param layer_names: list of tensor names to load, e.g. ["model.embed_tokens.weight", "lm_head.weight"]
    :param model_path: either a local directory containing model.safetensors.index
    :return: a single tensor or a dict of tensors
    """
    # download the index file
    index_file = _resolve_file(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index = json.load(f)

    weight_map: Dict[str, str] = index["weight_map"]

    # group requested names by shard filename
    shard_to_names: Dict[str, List[str]] = {}
    for name in layer_names:
        shard = weight_map.get(name)
        if shard is None:
            logger.warning(f"Tensor '{name}' not found in index weight_map.")
            continue
        shard_to_names.setdefault(shard, []).append(name)

    if not shard_to_names:
        raise ValueError("None of the requested tensor names were found in the index.")

    # fetch each required shard and extract only the requested tensors with safetensors.safe_open
    out: Dict[str, Any] = {}
    for shard_file, names in shard_to_names.items():
        shard_path = _resolve_file(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for name in names:
                if name not in available:
                    logger.warning(f"Tensor '{name}' not found inside shard '{shard_file}'.")
                    continue
                out[name] = f.get_tensor(name)

    if len(layer_names) == 1:
        return out.get(layer_names[0])
    return out

def _resolve_file(model_path: str, file_name: str) -> Path:
    """
    If model_path is a local directory, return path/<filename> if it exists.
    Otherwise treat model_path as a HF repo_id and download with hf_hub_download.

    :param model_path: local directory or HF repo_id
    :param file_name: filename to look for or download
    :return: local path to the resolved file
    """
    if os.path.isdir(model_path):
        logger.info("Loading from local directory:", model_path)
        p = os.path.join(model_path, file_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected local file missing: {p}")
        return p
    # Treat as repo_id on the Hub
    logger.info("Loading from huggingface directory:", model_path)
    return hf_hub_download(repo_id=model_path, filename=file_name)

if __name__ == "__main__":
    #model_path = "/home/hzhao/.cache/huggingface/hub/models--shanjiaz--Meta-Llama-3-8B-Instruct-FP8-BLOCK/snapshots/ea6d7c1a6a0874d9db6511ce93da2b777f24376f"
    model_path = "shanjiaz/Meta-Llama-3-8B-Instruct-FP8-BLOCK"
    layer_names = ["lm_head.weight", "model.embed_tokens.weight"]
    layer = load_model_layers(layer_names, model_path)
    for k, v in layer.items():
        print(k, v.shape)
        print(f"sample data: {v.flatten()[:5]}")
