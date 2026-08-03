#!/usr/bin/env python3
"""
Prepare data for speculator training

Accepted inputs contain target-model responses, either as natural-language
conversations or as prepared ``input_ids`` and ``loss_mask`` rows. Natural
language is rendered by the target model's vLLM endpoint; this command never
generates responses.

The output of this script is:
1. Processed dataset ready for online training or offline datagen in output_dir
2. Token frequency statistics file at token_freq_path

Preprocessing will be skipped if the dataset already exists at the output directory.
Token frequencies are saved in the output directory by default.

Usage:
    python prepare_data.py \
        --data ./on_policy_conversations.jsonl \
        --render-endpoint http://localhost:8000 \
        --output ./training_data \
        --max-samples 5000
"""

import argparse
import glob
import logging
import shutil
import sys
from pathlib import Path

from speculators.data_generation.preprocessing import (
    load_and_preprocess_dataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        action="append",
        required=True,
        help=(
            "On-policy conversations or prepared input_ids/loss_mask rows. "
            "Use local JSON/JSONL or hf:<dataset>; source presets belong to "
            "response regeneration."
        ),
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
        "--render-endpoint",
        type=str,
        default=None,
        help=(
            "Base URL of a running vLLM server (e.g. http://localhost:8000). "
            "The instance launched for hidden-state extraction serves this "
            "too, so no second server is needed. Pass the base URL only: "
            "/v1/chat/completions/render is appended to it, so the "
            "/v1-suffixed form that data_generation_offline.py --endpoint "
            "takes will 404. Conversations are tokenized by that endpoint and "
            "the loss mask is derived from the render boundary. Required "
            "unless every --data input already has input_ids and loss_mask."
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

    logger.info("Preparing %s into %s", args.data, args.output)

    output = Path(args.output)
    token_freq_path = (
        output / "token_freq.pt"
        if args.token_freq_path is None
        else Path(args.token_freq_path)
    )

    if output.exists():
        if not args.overwrite and glob.glob(str(output / "*.arrow")):
            logger.warning(
                "Dataset files already exists in output directory, skipping "
                "preprocessing. To existing overwrite files use --overwrite."
            )
            sys.exit(0)
        if args.overwrite:
            assert_safe_to_overwrite(output, token_freq_path)
            logger.warning("Removing existing output directory: %s", output)
            shutil.rmtree(output)
            output.mkdir(parents=True)
    else:
        output.mkdir(parents=True)

    dataset = load_and_preprocess_dataset(
        train_data_paths=args.data,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_preprocessing_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=token_freq_path,
        render_endpoint=args.render_endpoint,
        minimum_valid_tokens=args.minimum_valid_tokens,
        allow_empty_output=args.allow_empty_output,
    )

    logger.info("Writing dataset to %s", args.output)
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()
