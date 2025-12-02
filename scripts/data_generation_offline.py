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
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import config as datasets_config
from tqdm import tqdm  # type: ignore[import-untyped]

from speculators.data_generation.config_generator import DataGenerationConfig
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.preprocessing import load_and_preprocess_dataset
from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
)

# Constants
MAX_IO_WORKERS = 4  # Number of parallel file save operations

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
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (same as used in preprocessing, default: 2048)",
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


def find_last_checkpoint(output_dir: str) -> int:
    """Find the last successfully saved file index by scanning existing files."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    max_index = -1
    for file_path in output_path.iterdir():
        if file_path.name.startswith("data_") and file_path.name.endswith(".pt"):
            index_str = file_path.stem[5:]  # Remove "data_" prefix
            try:
                index = int(index_str)
                max_index = max(max_index, index)
            except ValueError:
                continue

    return max_index + 1


def save_sample_to_disk(data_dict, output_path):
    """Helper function to save a single sample to disk (for async execution)"""
    torch.save(data_dict, output_path)
    return output_path


def save_config(args, generator, num_samples, output_dir):
    """Save metadata config file for reproducibility."""
    log.subsection("Saving configuration metadata")

    cache_dir = (
        args.hf_cache_dir if args.hf_cache_dir else datasets_config.HF_DATASETS_CACHE
    )

    config = DataGenerationConfig.from_generator(
        generator=generator,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        cache_dir=str(cache_dir),
        num_samples=num_samples,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    config_path = Path(output_dir) / "data_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    log.info(f"Saved config v{config.version} to {config_path}")


def generate_and_save_hidden_states(args, dataset):
    """Generate hidden states and save each sample as a .pt file"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    start_file_idx = find_last_checkpoint(args.output_dir)
    if start_file_idx > 0:
        log.subsection(f"Resuming: {start_file_idx} files already exist")

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

    num_batches = (
        num_samples - start_sample_idx + args.batch_size - 1
    ) // args.batch_size

    # Use ThreadPoolExecutor for async file I/O
    max_io_workers = MAX_IO_WORKERS

    pbar = tqdm(
        range(start_sample_idx, num_samples, args.batch_size),
        desc="Generating hidden states",
        total=num_batches,
    )

    with ThreadPoolExecutor(max_workers=max_io_workers) as thread_executor:
        futures = []

        for i in pbar:
            batch_end = min(i + args.batch_size, num_samples)
            batch = dataset[i:batch_end]
            batch_input_ids = batch["input_ids"]
            batch_loss_mask = batch["loss_mask"]

            results = generator.generate(batch_input_ids)

            # Submit save operations to thread pool (async I/O)
            for j, result in enumerate(results):
                # Truncate loss_mask to match input_ids length (generator may truncate)
                input_len = len(result["input_ids"])
                loss_mask = batch_loss_mask[j][:input_len]

                result_cleaned = {
                    "input_ids": result["input_ids"],
                    "hidden_states": [h.contiguous() for h in result["hidden_states"]],
                    "loss_mask": loss_mask,
                }
                output_path = Path(args.output_dir) / f"data_{file_idx}.pt"
                future = thread_executor.submit(
                    save_sample_to_disk, result_cleaned, output_path
                )
                futures.append(future)
                file_idx += 1

        log.info("Waiting for remaining file saves to complete...")
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Saving files"
        ):
            future.result()

    samples_saved = file_idx - start_file_idx
    log.info(f"Saved {samples_saved} new data points to {args.output_dir}")

    save_config(args, generator, num_samples, args.output_dir)

    return samples_saved


def main():
    args = parse_args()

    log.section("EAGLE Offline Data Generation")
    log.config(
        {
            "Target Model": args.target_model_path,
            "Dataset": args.train_data_path,
            "Output Dir": args.output_dir,
            "Tensor Parallel": args.tensor_parallel_size,
            "Batch Size": args.batch_size,
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
    )
    num_saved = generate_and_save_hidden_states(args, dataset)

    log.section("Data generation complete!")
    log.info(f"Saved {num_saved} files to {args.output_dir}")


if __name__ == "__main__":
    main()
