#!/usr/bin/env python3
"""
Offline EAGLE Training Data Generation Pipeline

This script generates training data for EAGLE models by:
1. Automatically preprocessing data if needed (or loading from cache)
2. Using vLLM to extract hidden states from target model
3. Saving each data point as a separate .pt file

Preprocessing is cached automatically by HuggingFace datasets.
Token frequencies are saved in the current directory by default.

Usage:
    python data_generation_offline.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path sharegpt \
        --output-dir ./training_data \
        --hf-cache-dir /path/to/cache \
        --max-samples 5000
"""

import argparse
import logging

from speculators.data_generation.logging_utils import PipelineLogger  # noqa: E402
from speculators.data_generation.preprocessing import (  # noqa: E402
    load_and_preprocess_dataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = PipelineLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data offline")

    # Model arguments
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model",
    )

    # Data arguments
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (same as used in preprocessing)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for preprocessing and model (default: 2048)",
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
        default="./token_freq.pt",
        help="Path to save token frequency distribution (default: ./token_freq.pt)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for HuggingFace datasets cache. "
            "If not specified, uses HF_DATASETS_CACHE env var or default location. "
            "(default: None)"
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
        "--output-dir", type=str, required=True, help="Directory to save .pt files"
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
    return parser.parse_args()


def main():
    args = parse_args()

    log.section("EAGLE Offline Data Generation")
    log.config(
        {
            "Target Model": args.target_model_path,
            "Dataset": args.train_data_path,
            "Output Dir": args.output_dir,
        }
    )

    dataset, _ = load_and_preprocess_dataset(
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_preprocessing_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=args.token_freq_path,
        cache_dir=args.hf_cache_dir,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
    )


if __name__ == "__main__":
    main()
