#!/usr/bin/env python3
"""
Standalone script to preprocess raw chat data for EAGLE3 training.

This script tokenizes raw chats using the model's built-in chat template.
Preprocessed data is automatically cached by HuggingFace datasets.

Usage:
    # Basic usage
    python preprocess_data.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path sharegpt \
        --seq-length 2048 \
        --hf-cache-dir /path/to/cache \
        --build-dataset-num-proc 8

    # With turn dropout for data augmentation
    python preprocess_data.py \
        --target-model-path meta-llama/Llama-3.1-8B \
        --train-data-path sharegpt \
        --seq-length 2048 \
        --turn-dropout

    # With custom assistant pattern
    python preprocess_data.py \
        --target-model-path openai/gpt-oss-20b \
        --train-data-path sharegpt \
        --seq-length 2048 \
        --assistant-pattern \\
            '<\\|start\\|>assistant<\\|channel\\|>final<\\|message\\|>(.*?)<\\|return\\|>'
"""

import argparse
import logging

from speculators.data_generation.preprocessing import load_and_preprocess_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for EAGLE3 training")

    # Model paths
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model",
    )

    # Data paths
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (JSON/JSONL, sharegpt, or ultrachat)",
    )

    # Processing parameters
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for shuffling (default: 0)"
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
        "--build-dataset-num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset building (default: 8)",
    )

    # Optional flags
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to preprocess (default: None, process all)",
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

    return parser.parse_args()


def main():
    args = parse_args()

    preprocessed_dataset, tokenizer = load_and_preprocess_dataset(
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.build_dataset_num_proc,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=args.token_freq_path,
        cache_dir=args.hf_cache_dir,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
    )

    logger.info("Preprocessing complete!")
    if args.hf_cache_dir:
        logger.info(f"Preprocessed data cached at: {args.hf_cache_dir}")
    logger.info(f"Token frequencies saved at: {args.token_freq_path}")


if __name__ == "__main__":
    main()
