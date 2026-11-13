"""Quantize a drafter or speculator safetensors checkpoint with FP8."""

import argparse
from collections.abc import Sequence
from pathlib import Path

# Model-free PTQ reads safetensors without model class information; adapt these
# module-name regexes to your checkpoint. The defaults exclude DSpark modules.
IGNORE = (
    r"re:.*(^|\.)embed_tokens$",
    r"re:.*(^|\.)lm_head$",
    r"re:.*(^|\.)verifier_lm_head$",
    r"re:.*(^|\.)markov_head\..*",
    r"re:.*(^|\.)confidence_head\..*",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence; defaults to the process arguments.

    Returns:
        Parsed quantization options.
    """
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="Hub ID or local checkpoint")
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument(
        "--scheme",
        choices=["FP8_DYNAMIC", "FP8_BLOCK"],
        default="FP8_DYNAMIC",  # Change to "FP8_BLOCK" for blockwise FP8.
    )
    return cli.parse_args(args=argv)


def quantize(args: argparse.Namespace) -> None:
    """Quantize a checkpoint into a new output directory.

    Args:
        args: Parsed command-line arguments.
    """
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    from llmcompressor import model_free_ptq  # noqa: PLC0415

    model_free_ptq(
        model_stub=args.model,
        save_directory=str(args.output_dir),
        scheme=args.scheme,
        ignore=IGNORE,
    )


def main() -> None:
    """Run the FP8 quantization command-line interface."""
    quantize(args=parse_args())


if __name__ == "__main__":
    main()
