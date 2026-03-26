#!/usr/bin/env python3
"""Extract the MTP head from Qwen3-Next-80B-A3B-Instruct into speculators format.

Downloads the model from HuggingFace (or reads from a local cache), extracts the
MTP layer weights, remaps them to the speculators key convention, and writes a
self-contained speculators checkpoint directory containing config.json and
model.safetensors.

Key remapping (Qwen3-Next → speculators, inverse of stitch_weights.py):
    mtp.pre_fc_norm_hidden.weight    → mtp_layers.0.pre_fc_norm_hidden.weight
    mtp.pre_fc_norm_embedding.weight → mtp_layers.0.pre_fc_norm_embedding.weight
    mtp.fc.weight                    → mtp_layers.0.fc.weight
    mtp.norm.weight                  → mtp_layers.0.norm.weight
    mtp.layers.0.<key>               → mtp_layers.0.<key>

The remapping logic lives in :mod:`speculators.models.fast_mtp.checkpoint` and is
shared with stitch_weights.py to guarantee bidirectional consistency.

Usage:
    # From HuggingFace:
    python examples/fast_mtp/convert_checkpoint.py \\
        --output-dir Qwen3-Next-80B-A3B-Instruct_mtp_speculator

    # From a local snapshot:
    SNAP=9c7f2fbe84465e40164a94cc16cd30b6999b0cc7
    MODEL=/mnt/data/engine/hub_cache/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/$SNAP
    python examples/fast_mtp/convert_checkpoint.py \\
        --model $MODEL \\
        --output-dir Qwen3-Next-80B-A3B-Instruct_mtp_speculator
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

# Key remapping constants — single source of truth shared with stitch_weights.py.
# Importing private names from the same package is intentional here: these constants
# define the on-disk format and must stay in sync between the two directions.
from speculators.models.fast_mtp.checkpoint import (
    _MTP_TOP_LEVEL_SUFFIXES,  # noqa: PLC2701
    _SPECULATORS_PREFIX,  # noqa: PLC2701
)

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"

MTP_PREFIX = "mtp."

# Architecture fields to copy from the source config into transformer_layer_config
ARCH_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "intermediate_size",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "layer_types",
    "full_attention_interval",
    "head_dim",
    "attention_bias",
)

_MTP_LAYERS_PREFIX = "mtp.layers.0."


def remap_key(key: str) -> str:
    """Map a Qwen3-Next MTP key to the speculators ``mtp_layers.0.*`` convention.

    Inverse of :func:`speculators.models.fast_mtp.checkpoint.remap_key`.
    Uses the same frozenset constants for consistency — no separate KEY_REMAP dict.
    """
    if key.startswith(_MTP_LAYERS_PREFIX):
        # mtp.layers.0.<rest> → mtp_layers.0.<rest>
        return _SPECULATORS_PREFIX + key[len(_MTP_LAYERS_PREFIX) :]
    # mtp.<suffix> where suffix is a top-level mixin weight → mtp_layers.0.<suffix>
    # FastMTPLayerMixin attributes are named to match Qwen3-Next originals, so
    # the suffix is identical in both directions.
    suffix = key[len(MTP_PREFIX) :]
    if suffix in _MTP_TOP_LEVEL_SUFFIXES:
        return _SPECULATORS_PREFIX + suffix
    raise ValueError(f"Unexpected MTP key with no remapping rule: {key!r}")


def load_checkpoint(model: str, cache_dir: str | None) -> Path:
    if Path(model).exists():
        return Path(model)
    print(f"Downloading {model}...")
    return Path(
        snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
    )


def load_mtp_weights(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load only the shard(s) that contain MTP weights, using the index file."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        mtp_keys = [k for k in weight_map if k.startswith(MTP_PREFIX)]
        if not mtp_keys:
            raise ValueError(
                f"No MTP weights found with prefix {MTP_PREFIX!r}. "
                f"Sample keys: {list(weight_map)[:5]}"
            )
        shards = sorted({weight_map[k] for k in mtp_keys})
        print(f"Loading {len(shards)} shard(s) containing {len(mtp_keys)} MTP keys...")
        weights: dict[str, torch.Tensor] = {}
        for shard in shards:
            weights.update(load_file(checkpoint_dir / shard))
    else:
        # Single-file checkpoint
        shard_files = sorted(checkpoint_dir.glob("model*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
        print(f"Loading {len(shard_files)} shard(s)...")
        weights = {}
        for path in shard_files:
            weights.update(load_file(path))
        mtp_keys = [k for k in weights if k.startswith(MTP_PREFIX)]
        if not mtp_keys:
            raise ValueError(
                f"No MTP weights found with prefix {MTP_PREFIX!r}. "
                f"Sample keys: {list(weights)[:5]}"
            )

    mtp = {
        remap_key(k): v.clone() for k, v in weights.items() if k.startswith(MTP_PREFIX)
    }
    print(f"Extracted and remapped {len(mtp)} MTP tensors")
    preview = 5
    for k in sorted(mtp)[:preview]:
        print(f"  {k}: {list(mtp[k].shape)}")
    if len(mtp) > preview:
        print(f"  ... and {len(mtp) - preview} more")
    return mtp


def build_speculators_config(source_config: dict, model_id: str) -> dict:
    return {
        "speculators_model_type": "mtp",
        "architectures": ["FastMTPSpeculator"],
        "num_nextn_predict_layers": 1,
        "speculators_config": {
            "algorithm": "mtp",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 3,
                    "accept_tolerance": 0.0,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": ["Qwen3NextForCausalLM"],
                "name_or_path": model_id,
            },
        },
        "transformer_layer_config": {
            "model_type": "qwen3_next",
            **{k: source_config[k] for k in ARCH_FIELDS if k in source_config},
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert Qwen3-Next MTP head to speculators format"
    )
    p.add_argument(
        "--model",
        default=MODEL_ID,
        help="Local path or HF repo ID (default: %(default)s)",
    )
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    args = p.parse_args()

    checkpoint_dir = load_checkpoint(args.model, args.cache_dir)

    with (checkpoint_dir / "config.json").open() as f:
        source_config = json.load(f)

    mtp_weights = load_mtp_weights(checkpoint_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "model.safetensors"
    save_file(mtp_weights, str(weights_path))
    size_mb = weights_path.stat().st_size / 1e6
    print(f"Saved weights → {weights_path}  ({size_mb:.0f} MB)")

    config = build_speculators_config(source_config, args.model)
    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config  → {config_path}")


if __name__ == "__main__":
    main()
