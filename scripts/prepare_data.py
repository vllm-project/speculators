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
            "Forcibly rerun `prepare_data.py`.Deletes existing content in output dir"
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
    if output.exists():
        if not args.overwrite and glob.glob(str(output / "*.arrow")):
            log.warning(
                "Dataset files already exists in output directory, skipping "
                "preprocessing. To existing overwrite files use --overwrite."
            )
            sys.exit(0)
    else:
        output.mkdir(parents=True)

    token_freq_path = (
        output / "token_freq.pt"
        if args.token_freq_path is None
        else Path(args.token_freq_path)
    )

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
        trust_remote_code=args.trust_remote_code,
    )

    log.info("Done preparing data")
    log.section(f"Writing dataset to {args.output}")
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()
