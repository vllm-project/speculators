#!/usr/bin/env python3
"""
Offline EAGLE Training Data Generation Pipeline

This script generates training data for EAGLE models by:
1. Automatically preprocessing data if needed (or loading from cache)
2. Using vLLM to extract hidden states from target model
3. Saving each data point as a separate .pt file

The script intelligently handles preprocessing - if the data has already been
preprocessed with matching parameters, it loads from cache. Otherwise, it runs
preprocessing automatically. This eliminates the need to run preprocess_data.py
separately and prevents parameter mismatches.

Usage:
    # Basic usage - preprocessing happens automatically if needed
    python data_generation_offline.py \
        --target-model-path meta-llama/Llama-3.1-8B \
        --train-data-path sharegpt \
        --chat-template llama3 \
        --output-dir ./training_data \
        --max-samples 5000

    # Advanced usage with custom parameters
    python data_generation_offline.py \
        --target-model-path meta-llama/Llama-3.1-8B \
        --train-data-path sharegpt \
        --chat-template llama3 \
        --seq-length 2048 \
        --cache-dir ./cache \
        --output-dir ./training_data \
        --layer-ids 2 14 24 \
        --tensor-parallel-size 1 \
        --batch-size 8
"""

import argparse
import logging
import os

import torch
from datasets import load_from_disk
from tqdm import tqdm

from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.preprocessing import (
    generate_cache_key,
    load_and_preprocess_dataset,
)
from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for target model (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length supported by the model (default: 2048)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Target GPU memory utilization (default: 0.8)",
    )

    # Data arguments
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (same as used in preprocessing)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        required=True,
        help="Chat template name (same as used in preprocessing)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (same as used in preprocessing, default: 2048)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory where preprocessed data is cached (default: ./cache)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save .pt files"
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "List of layer IDs from which to capture hidden states "
            "(default: auto-select)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for hidden states generation (default: 8)",
    )

    # Processing arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (must match preprocessing seed, default: 0)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for output files (default: 0)",
    )
    parser.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=8,
        help="Number of CPU processes for dataset preprocessing (default: 8)",
    )

    return parser.parse_args()


def load_or_preprocess_dataset(args):
    """Load preprocessed dataset from cache, or run preprocessing if needed.

    This automatically handles preprocessing if the cached data doesn't exist,
    making the pipeline more user-friendly and preventing parameter mismatches.
    """
    # Generate cache key (must match the one used during preprocessing)
    cache_key = generate_cache_key(
        args.target_model_path,
        args.chat_template,
        args.seq_length,
        args.train_data_path,
    )

    if args.max_samples is not None:
        cache_key = f"{cache_key}_samples{args.max_samples}"

    dataset_cache_dir = os.path.join(args.cache_dir, "processed_dataset", cache_key)

    # Run preprocessing only when cached data is not found
    if os.path.exists(dataset_cache_dir):
        log.subsection("Loading cached preprocessed data")
        log.info(f"Cache: {dataset_cache_dir}")
        dataset = load_from_disk(dataset_cache_dir)
        log.info(f"Loaded {len(dataset)} samples")
    else:
        log.subsection(
            "Preprocessed data not found - running preprocessing automatically"
        )

        dataset, _ = load_and_preprocess_dataset(
            target_model_path=args.target_model_path,
            train_data_path=args.train_data_path,
            chat_template=args.chat_template,
            seq_length=args.seq_length,
            cache_dir=args.cache_dir,
            build_dataset_num_proc=args.num_preprocessing_workers,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        log.info(f"Data cached at: {dataset_cache_dir}")

    return dataset


def find_last_checkpoint(output_dir: str) -> int:
    """Find the last successfully saved file index by scanning existing files."""
    if not os.path.exists(output_dir):
        return 0

    existing_files = [
        f for f in os.listdir(output_dir) if f.startswith("data_") and f.endswith(".pt")
    ]
    if not existing_files:
        return 0

    # Extract indices and find max
    indices = [int(f.replace("data_", "").replace(".pt", "")) for f in existing_files]
    return max(indices) + 1


def generate_and_save_hidden_states(args, dataset):
    """Generate hidden states and save each sample as a .pt file"""

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-resume: find where we left off based on existing files
    start_file_idx = find_last_checkpoint(args.output_dir)
    if start_file_idx > 0:
        log.subsection(f"Resuming: {start_file_idx} files already exist")

    # Calculate which dataset samples to process
    num_samples = len(dataset)
    start_sample_idx = start_file_idx - args.start_idx

    if start_sample_idx >= num_samples:
        log.info("All samples already processed!")
        return 0

    log.subsection("Initializing vLLM hidden states generator")
    generator = VllmHiddenStatesGenerator(
        model_path=args.target_model_path,
        layer_ids=args.layer_ids,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    log.info(f"Processing {num_samples - start_sample_idx}/{num_samples} samples")
    file_idx = start_file_idx

    for i in tqdm(
        range(start_sample_idx, num_samples, args.batch_size),
        desc="Generating hidden states",
        initial=start_sample_idx,
        total=num_samples,
    ):
        batch_end = min(i + args.batch_size, num_samples)
        batch = dataset[i:batch_end]
        batch_input_ids = batch["input_ids"]
        batch_loss_mask = batch["loss_mask"]

        results = generator.generate(batch_input_ids)

        # Save each sample (one file per sample for variable-length sequences)
        for j, result in enumerate(results):
            result["loss_mask"] = batch_loss_mask[j]
            torch.save(result, os.path.join(args.output_dir, f"data_{file_idx}.pt"))
            file_idx += 1

    samples_saved = file_idx - start_file_idx
    log.info(f"Saved {samples_saved} new data points to {args.output_dir}")
    return samples_saved


def main():
    args = parse_args()

    log.section("EAGLE Offline Data Generation")
    log.config(
        {
            "Target Model": args.target_model_path,
            "Dataset": args.train_data_path,
            "Chat Template": args.chat_template,
            "Output Dir": args.output_dir,
            "Tensor Parallel": args.tensor_parallel_size,
            "Batch Size": args.batch_size,
        }
    )

    dataset = load_or_preprocess_dataset(args)
    num_saved = generate_and_save_hidden_states(args, dataset)

    log.section("Data generation complete!")
    log.info(f"Saved {num_saved} files to {args.output_dir}")


if __name__ == "__main__":
    main()
