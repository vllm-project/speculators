"""Stitch finetuned FastMTP weights back into the Qwen3-Next verifier.

After finetuning, the trained MTP head weights live in a speculators checkpoint.
This script extracts those weights, re-keys them back to the original Qwen3-Next
``mtp.*`` key namespace (the inverse of what convert_checkpoint.py does), and
writes a new model directory that vLLM can load directly.

The output directory contains:
  - Full copies of all original verifier safetensors shards
  - A new shard ``mtp_finetuned.safetensors`` with the updated MTP head weights
    stored under the original ``mtp.*`` key names that vLLM expects
  - An updated ``model.safetensors.index.json`` that routes ``mtp.*`` keys to
    the new shard (replacing the original shard's entries)
  - The verifier config.json (unchanged)

Key remapping is handled by :mod:`speculators.models.fast_mtp.checkpoint`.
Because the four :class:`FastMTPLayerMixin` attributes are named to match
Qwen3-Next originals exactly, no explicit rename dict is required — the namespace
is determined by frozenset membership alone.

Usage:
    python examples/fast_mtp/stitch_weights.py \\
        --checkpoint output/qwen3next_gsm8k_finetuned/best/model.safetensors \\
        --verifier /path/to/Qwen3-Next-80B-A3B-Instruct \\
        --output-dir output/stitched_model

Note: if the output directory already exists, existing files are skipped (the script
is idempotent). Use ``--overwrite`` to force replacement of existing files.

vLLM loads the full model from the output directory using the combined index.
"""

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from speculators.models.fast_mtp.checkpoint import remap_keys, update_weight_index


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
        help="Filename for the new MTP weight shard (default: %(default)s)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory. "
        "By default, existing files are skipped (idempotent).",
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
    preview = 5
    for k in sorted(remapped)[:preview]:
        print(f"    {k}")
    if len(remapped) > preview:
        print(f"    ... and {len(remapped) - preview} more")

    # Save MTP shard
    mtp_shard_path = output_dir / args.shard_name
    print(f"Saving MTP shard: {mtp_shard_path}")
    save_file(remapped, str(mtp_shard_path))

    # Copy all original verifier files into output directory
    print(f"Copying verifier files from {verifier_dir}")
    for src in sorted(verifier_dir.iterdir()):
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            if args.overwrite:
                print(f"  Overwriting: {src.name}")
            else:
                print(f"  Skipping (exists): {src.name}")
                continue
        shutil.copy2(src, dst)
        print(f"  Copied: {src.name}")

    # Update the weight index to route mtp.* keys to the new shard.
    # Raises FileNotFoundError if the verifier has no model.safetensors.index.json
    # (single-shard models are not supported).
    print("Updating model.safetensors.index.json")
    update_weight_index(
        verifier_dir=verifier_dir,
        output_dir=output_dir,
        mtp_keys=list(remapped.keys()),
        new_shard_name=args.shard_name,
    )

    print(f"\nStitched model saved to: {output_dir}")
    print("  To load with vLLM:")
    print(
        f"    llm = LLM(model='{output_dir}', "
        'speculative_config={"method": "mtp", "num_speculative_tokens": 3})'
    )


if __name__ == "__main__":
    main()
