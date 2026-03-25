#!/usr/bin/env python3
"""Extract the MTP head from Qwen3-Next-80B-A3B-Instruct into speculators format.

Downloads the model from HuggingFace (or reads from a local cache), extracts the
MTP layer weights, and writes a self-contained speculators checkpoint directory
containing config.json and model.safetensors.

Usage:
    python local/scripts/convert_checkpoints.py \
        --output-dir Qwen3-Next-80B-A3B-Instruct_mtp_speculator

    SNAP=9c7f2fbe84465e40164a94cc16cd30b6999b0cc7
    MODEL=/mnt/data/engine/hub_cache/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/$SNAP
    python local/scripts/convert_checkpoints.py \
        --output-dir Qwen3-Next-80B-A3B-Instruct_mtp_speculator \
        --model $MODEL
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# Key prefixes used by Qwen3-Next for MTP weights in the original checkpoint
MTP_PREFIXES = ("model.mtp_layers.0.",)


def load_checkpoint(model: str, cache_dir: str | None) -> Path:
    if Path(model).exists():
        return Path(model)
    print(f"Downloading {model}...")
    return Path(
        snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "config.json"],
        )
    )


def load_weights(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    shard_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
    print(f"Loading {len(shard_files)} shard(s)...")
    weights: dict[str, torch.Tensor] = {}
    for path in shard_files:
        weights.update(load_file(path))
    return weights


def extract_mtp_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mtp = {k: v.clone() for k, v in weights.items() if k.startswith(MTP_PREFIXES)}
    if not mtp:
        raise ValueError(
            f"No MTP weights found with prefixes {MTP_PREFIXES}. "
            f"Sample keys: {list(weights)[:5]}"
        )
    print(f"Extracted {len(mtp)} MTP tensors")
    for k in sorted(mtp)[:5]:
        print(f"  {k}: {list(mtp[k].shape)}")
    if len(mtp) > 5:
        print(f"  ... and {len(mtp) - 5} more")
    return mtp


def build_speculators_config(source_config: dict) -> dict:
    return {
        "speculators_model_type": "mtp",
        "architectures": ["FastMTPSpeculator"],
        "num_speculative_steps": 3,
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
                "name_or_path": MODEL_ID,
            },
        },
        "transformer_layer_config": {
            "model_type": "qwen3_next",
            **{
                k: source_config[k]
                for k in (
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
                if k in source_config
            },
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

    weights = load_weights(checkpoint_dir)
    mtp_weights = extract_mtp_weights(weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "model.safetensors"
    save_file(mtp_weights, str(weights_path))
    size_mb = weights_path.stat().st_size / 1e6
    print(f"Saved weights → {weights_path}  ({size_mb:.0f} MB)")

    config = build_speculators_config(source_config)
    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config  → {config_path}")


if __name__ == "__main__":
    main()
