#!/usr/bin/env python3
"""Extract the MTP head from Qwen3-Next-80B-A3B-Instruct into speculators format.

Downloads the model from HuggingFace (or reads from a local path), extracts the
``mtp.*`` weights, and writes a self-contained speculators checkpoint containing
``config.json`` and ``model.safetensors``.

Because the FastMTP ``MTPBlock`` layout mirrors the Qwen3-Next checkpoint
structure, extracted keys are stored as-is — no renaming is needed.

Usage:
    # From HuggingFace (downloads automatically):
    python examples/fast_mtp/convert_checkpoint.py \\
        --output-dir Qwen3-Next-80B-A3B-Instruct_mtp_speculator

    # From a local snapshot:
    MODEL=/path/to/Qwen3-Next-80B-A3B-Instruct
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
from transformers import AutoConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp import FastMTPConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
_MTP_PREFIX = "mtp."


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
    """Load only the shard(s) that contain MTP weights, using the index file.

    Keys are returned as-is from the checkpoint — no remapping needed.
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        mtp_keys = [k for k in weight_map if k.startswith(_MTP_PREFIX)]
        if not mtp_keys:
            raise ValueError(
                f"No MTP weights found with prefix {_MTP_PREFIX!r}. "
                f"Sample keys: {list(weight_map)[:5]}"
            )
        shards = sorted({weight_map[k] for k in mtp_keys})
        print(f"Loading {len(shards)} shard(s) containing {len(mtp_keys)} MTP keys...")
        weights: dict[str, torch.Tensor] = {}
        for shard in shards:
            weights.update(load_file(checkpoint_dir / shard))
    else:
        shard_files = sorted(checkpoint_dir.glob("model*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
        print(f"Loading {len(shard_files)} shard(s)...")
        weights = {}
        for path in shard_files:
            weights.update(load_file(path))
        mtp_keys = [k for k in weights if k.startswith(_MTP_PREFIX)]
        if not mtp_keys:
            raise ValueError(
                f"No MTP weights found with prefix {_MTP_PREFIX!r}. "
                f"Sample keys: {list(weights)[:5]}"
            )

    mtp = {k: v.clone() for k, v in weights.items() if k.startswith(_MTP_PREFIX)}
    print(f"Extracted {len(mtp)} MTP tensors")
    preview = 5
    for k in sorted(mtp)[:preview]:
        print(f"  {k}: {list(mtp[k].shape)}")
    if len(mtp) > preview:
        print(f"  ... and {len(mtp) - preview} more")
    return mtp


def build_speculators_config(checkpoint_dir: Path, model_id: str) -> FastMTPConfig:
    """Build a ``FastMTPConfig`` from the Qwen3-Next source config."""
    source_config = AutoConfig.from_pretrained(checkpoint_dir)
    return FastMTPConfig(
        transformer_layer_config=source_config,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            default_proposal_method="greedy",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=3),
            ],
            verifier=VerifierConfig.from_config(source_config, name_or_path=model_id),
        ),
    )


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

    mtp_weights = load_mtp_weights(checkpoint_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "model.safetensors"
    save_file(mtp_weights, str(weights_path))
    size_mb = weights_path.stat().st_size / 1e6
    print(f"Saved weights → {weights_path}  ({size_mb:.0f} MB)")

    config = build_speculators_config(checkpoint_dir, args.model)
    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved config  → {config_path}")


if __name__ == "__main__":
    main()
