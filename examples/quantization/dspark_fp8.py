"""Quantize a DSpark drafter with dynamic FP8 weights and activations."""

import argparse
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="DSpark Hub ID or local checkpoint")
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    from llmcompressor import model_free_ptq  # noqa: PLC0415

    # Replace FP8_DYNAMIC with FP8_BLOCK for blockwise FP8.
    model_free_ptq(
        model_stub=args.model,
        save_directory=str(args.output_dir),
        scheme="FP8_DYNAMIC",
        ignore=[
            r"re:.*(^|\.)embed_tokens$",
            "lm_head",
            "verifier_lm_head",
            r"re:.*(^|\.)markov_head\..*",
            r"re:.*(^|\.)confidence_head\..*",
        ],
    )


if __name__ == "__main__":
    main()
