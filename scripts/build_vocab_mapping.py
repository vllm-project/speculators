#!/usr/bin/env python3
"""
Build vocabulary mappings (d2t and t2d) from token frequency distribution.

This script takes a token frequency distribution file (generated during data
preprocessing) and creates vocabulary mappings for a draft model with a
smaller vocabulary.

Usage:
    python build_vocab_mapping.py \
        --token-freq-path ./cache/token_frequencies/xxx_token_freq.pt \
        --draft-vocab-size 32000 \
        --target-vocab-size 128256 \
        --output-path ./vocab_mapping
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build vocabulary mappings from token frequency distribution"
    )

    parser.add_argument(
        "--token-freq-path",
        type=str,
        required=True,
        help="Path to token frequency distribution file (.pt)",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        required=True,
        help="Vocabulary size for the draft model",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        required=False,
        help="Vocabulary size for the target model",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=False,
        help="Model name or path from which to load the target vocabulary",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the vocabulary mapping files",
    )

    return parser.parse_args()


def get_target_vocab_size(args):
    has_vocab = args.target_vocab_size is not None
    has_model = args.target_model_path is not None

    if has_vocab and has_model:
        raise ValueError("Cannot specify both target-vocab-size and target-model-path")

    if not has_vocab and not has_model:
        raise ValueError("Must specify either target-vocab-size or target-model-path")

    if has_vocab:
        return args.target_vocab_size

    logger.info(f"Loading target model config from {args.target_model_path}")
    config = AutoConfig.from_pretrained(args.target_model_path)
    return config.vocab_size


def main():
    args = parse_args()

    token_freq_path = Path(args.token_freq_path)
    if not token_freq_path.exists():
        raise FileNotFoundError(f"Token frequency file not found: {token_freq_path}")

    token_freq_dict = torch.load(token_freq_path, weights_only=True)

    target_vocab_size = get_target_vocab_size(args)

    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as .npy files (expected by training script)
    d2t_path = output_path / "d2t.npy"
    t2d_path = output_path / "t2d.npy"

    np.save(d2t_path, d2t.cpu().numpy())
    np.save(t2d_path, t2d.cpu().numpy())

    logger.info(f"Saved d2t to {d2t_path} (shape: {d2t.shape})")
    logger.info(f"Saved t2d to {t2d_path} (shape: {t2d.shape})")


if __name__ == "__main__":
    main()
