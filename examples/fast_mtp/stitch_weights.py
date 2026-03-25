"""Stitch finetuned FastMTP weights back into the Qwen3-Next verifier.

After finetuning, the trained MTP head weights live in a speculators checkpoint.
This script extracts those weights, re-keys them to match Qwen3-Next's native
``model.mtp_layers.0.*`` key namespace, and writes a new model directory that
vLLM can load directly alongside the original verifier weights.

The output directory contains:
  - Symlinks to all original verifier safetensors shards (no copying)
  - A new shard ``mtp_finetuned.safetensors`` with the updated MTP head weights
  - An updated ``model.safetensors.index.json`` that routes MTP keys to the new shard
  - The verifier config.json (unchanged)

vLLM loads the full model from this directory using the combined index.

Usage:
    python examples/fast_mtp/stitch_weights.py \\
        --checkpoint output/qwen3next_gsm8k_finetuned/best/model.safetensors \\
        --verifier /path/to/Qwen3-Next-80B-A3B-Instruct \\
        --output-dir output/stitched_model
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

# Speculators checkpoint key prefix -> Qwen3-Next native prefix
# Speculators saves: mtp_layers.0.{module}.{param}
# vLLM/Qwen3-Next expects: model.mtp_layers.0.{module}.{param}
SPECULATORS_PREFIX = "mtp_layers."
QWEN3_NEXT_PREFIX = "model.mtp_layers."

# Keys to skip when writing MTP shard (frozen weights in speculator checkpoint
# that should stay in the verifier's own shards)
_SKIP_KEYS = {"embed_tokens.weight", "lm_head.weight"}


def remap_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap speculators checkpoint keys to Qwen3-Next native format.

    Drops embed_tokens and lm_head (these belong to the verifier's shards).
    All mtp_layers.* keys are prefixed with ``model.``.
    """
    remapped = {}
    for key, tensor in state_dict.items():
        # Skip frozen verifier weights stored in the speculator checkpoint
        if any(key.startswith(skip) for skip in _SKIP_KEYS):
            continue
        if key.startswith(SPECULATORS_PREFIX):
            new_key = QWEN3_NEXT_PREFIX + key[len(SPECULATORS_PREFIX) :]
        else:
            # rotary_emb or other top-level keys — prefix with model.
            new_key = "model." + key
        remapped[new_key] = tensor
    return remapped


# ---------------------------------------------------------------------------
# Index update
# ---------------------------------------------------------------------------


def update_weight_index(
    verifier_dir: Path,
    output_dir: Path,
    mtp_keys: list[str],
    new_shard_name: str,
) -> None:
    """Copy and update the verifier's safetensors index to include MTP shard.

    Removes any existing MTP key entries (so we override them with finetuned
    weights) and adds the new shard for all remapped MTP keys.
    """
    index_path = verifier_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file model (unlikely for 80B but handle it)
        print("  No index file found; verifier may be a single-shard model.")
        return

    with index_path.open() as f:
        index = json.load(f)

    weight_map: dict[str, str] = index.get("weight_map", {})

    # Remove stale MTP entries from the original index
    for key in list(weight_map.keys()):
        if "mtp_layers" in key:
            del weight_map[key]

    # Add new MTP entries pointing to our finetuned shard
    for key in mtp_keys:
        weight_map[key] = new_shard_name

    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stitch finetuned MTP weights into verifier"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the finetuned speculator safetensors file "
        "(e.g., output/best/model.safetensors)",
    )
    p.add_argument(
        "--verifier",
        required=True,
        help="Path to the original Qwen3-Next-80B-A3B-Instruct model directory",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the stitched model",
    )
    p.add_argument(
        "--shard-name",
        default="mtp_finetuned.safetensors",
        help="Filename for the new MTP weight shard",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    verifier_dir = Path(args.verifier)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not verifier_dir.exists():
        raise FileNotFoundError(f"Verifier directory not found: {verifier_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load finetuned speculator weights
    print(f"Loading finetuned checkpoint: {checkpoint_path}")
    state_dict = load_file(str(checkpoint_path))
    print(f"  Keys in checkpoint: {len(state_dict)}")

    # Remap keys to Qwen3-Next namespace
    remapped = remap_keys(state_dict)
    print(f"  MTP keys after remapping: {len(remapped)}")
    for k in sorted(remapped)[:5]:
        print(f"    {k}")
    if len(remapped) > 5:
        print(f"    ... and {len(remapped) - 5} more")

    # Save MTP shard
    mtp_shard_path = output_dir / args.shard_name
    print(f"Saving MTP shard: {mtp_shard_path}")
    save_file(remapped, str(mtp_shard_path))

    # Symlink all original verifier files into output directory
    print(f"Symlinking verifier files from {verifier_dir}")
    for src in verifier_dir.iterdir():
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue  # don't overwrite the MTP shard or existing files
        dst.symlink_to(src.resolve())

    # Update the weight index to include the new MTP shard
    update_weight_index(
        verifier_dir=verifier_dir,
        output_dir=output_dir,
        mtp_keys=list(remapped.keys()),
        new_shard_name=args.shard_name,
    )

    # Copy verifier config (the stitched model is still a Qwen3-Next model)
    verifier_config_src = verifier_dir / "config.json"
    if verifier_config_src.exists():
        dst_config = output_dir / "config.json"
        if dst_config.is_symlink():
            dst_config.unlink()
        shutil.copy2(verifier_config_src, dst_config)

    print(f"\nStitched model saved to: {output_dir}")
    print("  To load with vLLM:")
    print(f"    llm = LLM(model='{output_dir}', speculative_model='<mtp>', ...)")


if __name__ == "__main__":
    main()
