"""Stitch finetuned FastMTP weights back into the Qwen3-Next verifier.

After finetuning, the trained MTP head weights live in a speculators checkpoint.
This script extracts those weights and writes a new model directory that vLLM
can load directly.

The output directory contains:
  - Full copies of all original verifier safetensors shards
  - A new shard ``mtp_finetuned.safetensors`` with the updated MTP head weights
  - An updated ``model.safetensors.index.json`` routing ``mtp.*`` keys to
    the new shard (replacing the original shard's entries)
  - The verifier ``config.json`` (unchanged)

No key renaming is needed: the ``MTPBlock`` module layout produces ``mtp.*``
keys that already match vLLM's expected format.  The only transformation is
filtering out the frozen verifier weights (``embed_tokens.weight``,
``lm_head.weight``) that are present in training checkpoints but already
covered by the verifier shards.

Usage:
    python examples/fast_mtp/stitch_weights.py \\
        --checkpoint output/qwen3next_gsm8k_finetuned/best/model.safetensors \\
        --verifier /path/to/Qwen3-Next-80B-A3B-Instruct \\
        --output-dir output/stitched_model

Note: existing files are skipped by default (idempotent). Use ``--overwrite``
to force replacement.
"""

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from speculators.models.fast_mtp.checkpoint import filter_mtp_keys, update_weight_index


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

    print(f"Loading finetuned checkpoint: {checkpoint_path}")
    state_dict = load_file(str(checkpoint_path))
    print(f"  Keys in checkpoint: {len(state_dict)}")

    mtp_weights = filter_mtp_keys(state_dict)
    print(f"  MTP keys (verifier weights excluded): {len(mtp_weights)}")
    preview = 5
    for k in sorted(mtp_weights)[:preview]:
        print(f"    {k}")
    if len(mtp_weights) > preview:
        print(f"    ... and {len(mtp_weights) - preview} more")

    mtp_shard_path = output_dir / args.shard_name
    print(f"Saving MTP shard: {mtp_shard_path}")
    save_file(mtp_weights, str(mtp_shard_path))

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

    print("Updating model.safetensors.index.json")
    update_weight_index(
        verifier_dir=verifier_dir,
        output_dir=output_dir,
        mtp_keys=list(mtp_weights.keys()),
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
