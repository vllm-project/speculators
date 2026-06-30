#!/usr/bin/env python3
"""
Prepare data for speculator training

This script processes an input dataset and:
1. Applies chat template + tokenizes each sample
2. Produces a loss/assistant mask for each sample
3. Records token frequency statistics

The output of this script is:
1. Processed dataset ready for online training or offline datagen in output_dir
2. Token frequency statistics file at token_freq_path

Preprocessing will be skipped if the dataset already exists at the output directory.
Token frequencies are saved in the output directory by default.

Usage:
    python prepare_data.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import argparse
import glob
import logging
import shutil
import sys
from pathlib import Path

from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.preprocessing import (
    load_and_preprocess_dataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = PipelineLogger(__name__)


# Files prepare_data.py itself writes into --output; only these may be removed by
# --overwrite.
PREPARE_DATA_OVERWRITE_ALLOWED_FILES = {
    "dataset_info.json",
    "state.json",
    "token_freq.pt",
}


def assert_safe_to_overwrite(output: Path, token_freq_path: Path) -> None:
    """Refuse to ``--overwrite`` a directory holding non-artifact files.

    Guards against pointing ``--output`` at a directory with unrelated user files
    and wiping it: only prepare_data.py's own outputs (``.arrow`` shards, dataset
    metadata, and the token frequency file) may be deleted.
    """
    unexpected_paths = []
    resolved_token_freq_path = token_freq_path.resolve()
    for path in output.iterdir():
        if path.is_file() and (
            path.suffix == ".arrow"
            or path.name in PREPARE_DATA_OVERWRITE_ALLOWED_FILES
            or path.resolve() == resolved_token_freq_path
        ):
            continue
        unexpected_paths.append(path)

    if unexpected_paths:
        formatted_paths = ", ".join(str(path) for path in unexpected_paths)
        raise ValueError(
            "--overwrite would delete files that do not look like prepare_data.py "
            f"artifacts: {formatted_paths}. Remove them manually or choose a "
            "different --output directory."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for speculator training")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow executing code from HF Hub when loading the target model's "
            "processor."
        ),
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        action="append",
        required=True,
        help="Path to training data (same as used in preprocessing)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for preprocessing and model (default: 8192)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )
    parser.add_argument(
        "--token-freq-path",
        type=str,
        default=None,
        help=(
            "Path to save token frequency distribution "
            "(default: args.output / 'token_freq.pt')"
        ),
    )
    parser.add_argument(
        "--assistant-pattern",
        type=str,
        default=None,
        help=(
            "Custom regex pattern for matching assistant responses. "
            "If not provided, auto-detected from chat template."
        ),
    )
    parser.add_argument(
        "--turn-dropout",
        action="store_true",
        help=(
            "Enable turn dropout: randomly keeps first N consecutive turns "
            "per conversation for data augmentation."
        ),
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to save output dataset (default: ./output)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Forcibly rerun `prepare_data.py`. Deletes existing content in output dir"
        ),
    )

    # Processing arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (must match preprocessing seed, default: 0)",
    )
    parser.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=8,
        help="Number of CPU processes for dataset preprocessing (default: 8)",
    )
    parser.add_argument(
        "--minimum-valid-tokens",
        type=int,
        default=None,
        help=(
            "Drop samples whose loss mask contains fewer than this many "
            "trainable tokens."
        ),
    )
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help=(
            "Allow writing an empty preprocessed dataset. By default prepare_data.py "
            "raises when normalization or filtering removes every sample."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.section("Preparing data")
    log.config(
        {
            "Target Model": args.model,
            "Dataset": args.data,
            "Output Dir": args.output,
        }
    )

    output = Path(args.output)
    token_freq_path = (
        output / "token_freq.pt"
        if args.token_freq_path is None
        else Path(args.token_freq_path)
    )

    if output.exists():
        if not args.overwrite and glob.glob(str(output / "*.arrow")):
            log.warning(
                "Dataset files already exists in output directory, skipping "
                "preprocessing. To existing overwrite files use --overwrite."
            )
            sys.exit(0)
        if args.overwrite:
            assert_safe_to_overwrite(output, token_freq_path)
            log.warning(f"Removing existing output directory: {output}")
            shutil.rmtree(output)
            output.mkdir(parents=True)
    else:
        output.mkdir(parents=True)

    dataset, _ = load_and_preprocess_dataset(
        target_model_path=args.model,
        train_data_paths=args.data,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_preprocessing_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=token_freq_path,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
        minimum_valid_tokens=args.minimum_valid_tokens,
        allow_empty_output=args.allow_empty_output,
        trust_remote_code=args.trust_remote_code,
    )

    log.info("Done preparing data")
    log.section(f"Writing dataset to {args.output}")
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()
