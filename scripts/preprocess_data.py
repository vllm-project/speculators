#!/usr/bin/env python3
"""
Standalone script to preprocess raw chat data for EAGLE3 training.

This script tokenizes raw chats, applies chat templates, and builds EAGLE3 datasets.

Usage:
    python preprocess_data.py \
        --target-model-path meta-llama/Llama-3.1-8B \
        --train-data-path sharegpt \
        --chat-template qwen2 \
        --seq-length 2048 \
        --cache-dir ./cache \
        --build-dataset-num-proc 8
"""

import argparse
import logging

from speculators.data_generation.preprocessing import (
    generate_cache_key,
    load_and_preprocess_dataset,
    view_samples,
)

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
        "--chat-template",
        type=str,
        required=True,
        help="Chat template name (e.g., qwen2, llama3, etc.)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for caching processed data (default: ./cache)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for shuffling (default: 0)"
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
        "--view-samples",
        type=int,
        default=0,
        help="Number of samples to view for sanity check (default: 0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cache_key = generate_cache_key(
        args.target_model_path,
        args.chat_template,
        args.seq_length,
        args.train_data_path,
    )

    preprocessed_dataset, tokenizer = load_and_preprocess_dataset(
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        chat_template=args.chat_template,
        seq_length=args.seq_length,
        cache_dir=args.cache_dir,
        build_dataset_num_proc=args.build_dataset_num_proc,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if args.view_samples > 0:
        view_samples(preprocessed_dataset, tokenizer, args.view_samples)

    logger.info(f"Dataset: {args.cache_dir}/processed_dataset/{cache_key}")
    logger.info(
        f"Token frequencies: {args.cache_dir}/token_frequencies/"
        f"{cache_key}_token_freq.pt"
    )


if __name__ == "__main__":
    main()
