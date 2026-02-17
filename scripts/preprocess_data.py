#!/usr/bin/env python3
"""
Preprocess dataset for EAGLE online training.

Tokenizes conversations, computes token frequencies and sample lengths.
These metadata files are used by Eagle3OnlineVLLMDataset for accurate
batch packing and vocabulary mapping.

Output structure:
    {output_dir}/
        token_freq.pt          # dict[int, int] mapping token_id -> count
        sample_lengths.json    # [len_0, len_1, ..., len_{N-1}]

Usage:
    python preprocess_data.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path sharegpt \
        --output-dir ./preprocessed_data \
        --max-samples 5000
"""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm  # type: ignore[import-untyped]

from speculators.data_generation.preprocessing import load_and_preprocess_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def save_sample_lengths(dataset, output_dir: str) -> None:
    """Save sample_lengths.json as a JSON list of tokenized lengths."""
    sample_lengths = [
        len(dataset[idx]["input_ids"])
        for idx in tqdm(range(len(dataset)), desc="Computing sample lengths")
    ]

    lengths_path = Path(output_dir) / "sample_lengths.json"
    with open(lengths_path, "w") as f:
        json.dump(sample_lengths, f)

    log.info(f"Saved {len(sample_lengths)} sample lengths to {lengths_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for EAGLE online training"
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (used for tokenizer)",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Dataset name ('sharegpt', 'ultrachat') or path to JSON/JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save all output files",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset shuffling (default: 0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CPU processes for dataset building (default: 8)",
    )
    parser.add_argument(
        "--assistant-pattern",
        type=str,
        default=None,
        help="Custom regex pattern for matching assistant responses",
    )
    parser.add_argument(
        "--turn-dropout",
        action="store_true",
        help="Enable random turn dropout during preprocessing",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Without this flag, the script "
        "skips processing if output files already exist.",
    )
    return parser.parse_args()


def _outputs_exist(output_dir: str) -> bool:
    """Check if both expected output files already exist."""
    output_path = Path(output_dir)
    token_freq = output_path / "token_freq.pt"
    sample_lengths = output_path / "sample_lengths.json"
    return token_freq.exists() and sample_lengths.exists()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_dir)

    if _outputs_exist(args.output_dir) and not args.overwrite:
        log.info(
            f"Output files already exist in {args.output_dir}. "
            "Skipping preprocessing. Use --overwrite to regenerate."
        )
        return

    output_path.mkdir(parents=True, exist_ok=True)
    token_freq_path = str(output_path / "token_freq.pt")

    log.info("Starting dataset preprocessing")
    log.info(f"  Target model: {args.target_model_path}")
    log.info(f"  Dataset: {args.train_data_path}")
    log.info(f"  Output dir: {args.output_dir}")
    log.info(f"  Seq length: {args.seq_length}")
    if args.max_samples:
        log.info(f"  Max samples: {args.max_samples}")

    # Step 1: Tokenize, build loss masks, and compute token frequencies
    dataset, _tokenizer = load_and_preprocess_dataset(
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=token_freq_path,
        cache_dir=args.hf_cache_dir,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
    )

    # Step 2: Save sample lengths
    save_sample_lengths(dataset, args.output_dir)

    log.info("Preprocessing complete!")
    log.info(f"  token_freq.pt")
    log.info(f"  sample_lengths.json ({len(dataset)} samples)")


if __name__ == "__main__":
    main()
