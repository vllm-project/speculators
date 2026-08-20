import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger
from safetensors import safe_open

_WEIGHT_ALIASES: dict[str, list[str]] = {
    "embed_tokens.weight": ["tok_embeddings.weight", "llm.embed.weight"],
    "lm_head.weight": ["output.weight", "llm.unembed.weight"],
    "model.norm.weight": ["norm.weight"],
}

_MODELOPT_QUANT_CONFIG = "hf_quant_config.json"
_W4A16_NVFP4 = "W4A16_NVFP4"
_NVFP4_GROUP_SIZE = 16
_MATRIX_DIMENSIONS = 2


def _resolve_key(name: str, weight_map: dict[str, str]) -> str | None:
    """Try exact match, then suffix match, then known aliases."""
    for candidate in [name, *_WEIGHT_ALIASES.get(name, [])]:
        if candidate in weight_map:
            return candidate
        matched = next((k for k in weight_map if k.endswith(candidate)), None)
        if matched:
            return matched
    return None


def is_config_only_dir(path: str | Path) -> bool:
    """Return True if ``path`` is a local directory with a ``config.json`` but no
    weight files (``*.safetensors`` / ``*.bin``).

    Used to distinguish a saved speculator *config* (from which a fresh draft is
    initialized) from a full checkpoint whose weights should be loaded.

    :param path: A local directory path. Hub ids and non-directories return False.
    :return: True when the directory holds a config but no weights.
    """
    directory = Path(path)
    if not directory.is_dir():
        return False
    has_config = (directory / "config.json").is_file()
    # Weight files, plus sharded-checkpoint index files (e.g.
    # model.safetensors.index.json) -- the latter end in .json and would not match
    # the *.safetensors / *.bin globs, so a shard manifest must be checked explicitly
    # to avoid treating an incomplete sharded checkpoint as config-only.
    has_weights = (
        any(directory.glob("*.safetensors"))
        or any(directory.glob("*.bin"))
        or any(directory.glob("*.safetensors.index.json"))
        or any(directory.glob("*.bin.index.json"))
    )
    return has_config and not has_weights


def list_checkpoint_keys(checkpoint_dir: str | Path) -> list[str]:
    """List all tensor keys in a checkpoint without loading weights.

    Supports sharded safetensors (via index) and single safetensors formats.

    :param checkpoint_dir: Path to a local checkpoint directory.
    :return: List of tensor key names present in the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            return list(json.load(f)["weight_map"].keys())

    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return list(f.keys())

    raise FileNotFoundError(
        f"No safetensors checkpoint found at {checkpoint_dir}. "
        "Expected model.safetensors.index.json or model.safetensors."
    )


def load_model_layers(
    layer_names: list[str],
    model_path: str,
    *,
    row_indices: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Load one or more named tensors from a HF repo using safetensors shards.
    Supports both exact keys and suffix pattern matching. Packed ModelOpt
    W4A16 NVFP4 weights are materialized to BF16 when their checkpoint metadata
    declares the verified unswizzled layout.

    :param layer_names: list of tensor names or suffix patterns to load, e.g.
    ["model.embed_tokens.weight", "lm_head.weight"]
    :param model_path: either a local directory of huggingface model
    containing model.safetensors.index
    :param row_indices: optional mapping from a requested layer name to a boolean
    mask or integer indices applied along its first dimension. Row-shaped NVFP4
    metadata is selected before dequantization to bound peak memory.
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

    # Resolve names: try exact match, then suffix match, then known aliases
    name_to_key = {}  # Maps input name to actual checkpoint key
    for name in layer_names:
        key = _resolve_key(name, weight_map)
        if key:
            name_to_key[name] = key
        else:
            logger.warning(f"Tensor '{name}' not found in weight_map.")

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
                tensor = f.get_tensor(key)
                selector = (row_indices or {}).get(name)
                if tensor.dtype == torch.uint8:
                    out[name] = _materialize_modelopt_w4a16_nvfp4_weight(
                        tensor,
                        key=key,
                        model_path=model_path,
                        weight_map=weight_map,
                        row_indices=selector,
                    )
                else:
                    out[name] = _select_rows(tensor, selector, name)
    return out


def _select_rows(
    tensor: torch.Tensor,
    row_indices: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    """Apply a validated CPU row selector without changing its order."""
    if row_indices is None:
        return tensor
    if tensor.ndim == 0:
        raise ValueError(f"Row indexing is invalid for scalar tensor '{name}'.")

    selector = row_indices.detach().to(device="cpu")
    if selector.ndim != 1:
        raise ValueError(
            f"Row selector for '{name}' must be 1-D, got {tuple(selector.shape)}."
        )
    if selector.dtype == torch.bool:
        if selector.numel() != tensor.shape[0]:
            raise ValueError(
                f"Boolean row selector for '{name}' has shape "
                f"{tuple(selector.shape)}, expected ({tensor.shape[0]},)."
            )
    elif selector.dtype in {torch.int32, torch.int64}:
        if selector.numel() and (
            selector.min().item() < 0 or selector.max().item() >= tensor.shape[0]
        ):
            raise IndexError(
                f"Row selector for '{name}' contains an index outside "
                f"[0, {tensor.shape[0]})."
            )
    else:
        raise TypeError(
            f"Row selector for '{name}' must be boolean or integer, "
            f"got {selector.dtype}."
        )
    return tensor[selector]


def _load_checkpoint_tensor(
    model_path: str,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    """Load one exact checkpoint key from its mapped safetensors shard."""
    shard_path = _resolve_file(model_path, weight_map[key])
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _modelopt_layer_quantization(
    model_path: str,
    weight_key: str,
) -> dict[str, Any]:
    """Return the unambiguous ModelOpt quantized-layer entry for a weight."""
    try:
        config_path = _resolve_file(model_path, _MODELOPT_QUANT_CONFIG)
    except (FileNotFoundError, EntryNotFoundError) as error:
        raise ValueError(
            f"Packed uint8 tensor '{weight_key}' requires "
            f"{_MODELOPT_QUANT_CONFIG} to identify its quantization format."
        ) from error

    with config_path.open() as config_file:
        config = json.load(config_file)
    producer = config.get("producer", {})
    producer_name = producer.get("name") if isinstance(producer, dict) else None
    if not isinstance(producer_name, str) or producer_name.lower() != "modelopt":
        raise ValueError(
            f"Packed uint8 tensor '{weight_key}' requires a ModelOpt-produced "
            f"{_MODELOPT_QUANT_CONFIG}, got producer {producer_name!r}."
        )
    quantization = config.get("quantization", {})
    quantized_layers = quantization.get("quantized_layers", {})
    if not isinstance(quantized_layers, dict):
        raise ValueError(
            f"Invalid {_MODELOPT_QUANT_CONFIG}: 'quantized_layers' must be a mapping."
        )

    layer_name = weight_key.removesuffix(".weight")
    matches = [
        entry
        for name, entry in quantized_layers.items()
        if name == layer_name
        or layer_name.endswith(f".{name}")
        or name.endswith(f".{layer_name}")
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(
            f"Packed uint8 tensor '{weight_key}' must have exactly one matching "
            f"entry in {_MODELOPT_QUANT_CONFIG}; found {len(matches)}."
        )
    return matches[0]


def _materialize_modelopt_w4a16_nvfp4_weight(
    packed_weight: torch.Tensor,
    *,
    key: str,
    model_path: str,
    weight_map: dict[str, str],
    row_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Materialize the verified unswizzled ModelOpt W4A16 NVFP4 layout.

    The layout stores two E2M1 values in each byte, one row-major E4M3 scale per
    16 input elements, and one scalar FP32 global scale. W4A16 does not quantize
    activations, so a dense frozen projection retains input-gradient semantics.
    """
    if not key.endswith(".weight"):
        raise ValueError(
            f"Unsupported packed uint8 tensor '{key}': expected a weight tensor."
        )

    layer_quantization = _modelopt_layer_quantization(model_path, key)
    quant_algo = layer_quantization.get("quant_algo")
    group_size = layer_quantization.get("group_size")
    if quant_algo != _W4A16_NVFP4 or group_size != _NVFP4_GROUP_SIZE:
        raise ValueError(
            f"Packed uint8 tensor '{key}' uses quant_algo={quant_algo!r}, "
            f"group_size={group_size!r}; only the verified unswizzled "
            f"{_W4A16_NVFP4} group-size {_NVFP4_GROUP_SIZE} layout is supported."
        )

    prefix = key.removesuffix(".weight")
    scale_key = f"{prefix}.weight_scale"
    global_scale_key = f"{prefix}.weight_scale_2"
    missing = [
        scale_name
        for scale_name in (scale_key, global_scale_key)
        if scale_name not in weight_map
    ]
    if missing:
        raise ValueError(
            f"Packed uint8 tensor '{key}' is missing ModelOpt NVFP4 metadata {missing}."
        )

    block_scales = _load_checkpoint_tensor(model_path, weight_map, scale_key)
    global_scale = _load_checkpoint_tensor(model_path, weight_map, global_scale_key)
    if (
        packed_weight.ndim != _MATRIX_DIMENSIONS
        or block_scales.ndim != _MATRIX_DIMENSIONS
    ):
        raise ValueError(
            f"ModelOpt NVFP4 '{key}' expects 2-D weight/scales, got "
            f"{tuple(packed_weight.shape)} and {tuple(block_scales.shape)}."
        )
    if block_scales.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"ModelOpt NVFP4 '{scale_key}' must use float8_e4m3fn, got "
            f"{block_scales.dtype}."
        )
    if global_scale.dtype != torch.float32 or global_scale.ndim != 0:
        raise ValueError(
            f"ModelOpt NVFP4 '{global_scale_key}' must be a scalar float32, got "
            f"dtype {global_scale.dtype} and shape {tuple(global_scale.shape)}."
        )
    if block_scales.shape[0] != packed_weight.shape[0]:
        raise ValueError(
            f"ModelOpt NVFP4 '{key}' row mismatch: weight has "
            f"{packed_weight.shape[0]}, scales have {block_scales.shape[0]}."
        )

    hidden_size = packed_weight.shape[1] * 2
    expected_scale_columns = hidden_size // _NVFP4_GROUP_SIZE
    if (
        hidden_size % _NVFP4_GROUP_SIZE
        or block_scales.shape[1] != expected_scale_columns
    ):
        raise ValueError(
            f"ModelOpt NVFP4 '{key}' has incompatible packed/scales shapes: "
            f"{tuple(packed_weight.shape)} and {tuple(block_scales.shape)}."
        )

    # Apply the selector to every row-shaped input before allocating the dense
    # output. The global scale is scalar and intentionally remains unchanged.
    packed_weight = _select_rows(packed_weight, row_indices, key)
    block_scales = _select_rows(block_scales, row_indices, scale_key)

    logger.info(
        "Materializing unswizzled ModelOpt W4A16 NVFP4 tensor '{}' as BF16 with "
        "shape ({}, {}).",
        key,
        packed_weight.shape[0],
        hidden_size,
    )
    return _dequantize_unswizzled_nvfp4(
        packed_weight,
        block_scales,
        global_scale,
    )


def _dequantize_unswizzled_nvfp4(
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """CPU, row-chunked equivalent of vLLM's unswizzled NVFP4 dequantizer."""
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}.")
    rows, packed_hidden_size = packed_weight.shape
    hidden_size = packed_hidden_size * 2
    output = torch.empty((rows, hidden_size), dtype=torch.bfloat16)
    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    global_scale_f32 = global_scale.to(torch.float32)

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        packed = packed_weight[start:end]
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        nibbles = torch.stack((low, high), dim=-1).reshape(end - start, hidden_size)
        magnitudes = e2m1[(nibbles & 0x07).to(torch.long)]
        signs = torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
        values = (magnitudes * signs).reshape(
            end - start,
            hidden_size // _NVFP4_GROUP_SIZE,
            _NVFP4_GROUP_SIZE,
        )
        scales = block_scales[start:end].to(torch.float32) * global_scale_f32
        output[start:end] = (values * scales.unsqueeze(-1)).reshape(
            end - start, hidden_size
        )

    return output


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
