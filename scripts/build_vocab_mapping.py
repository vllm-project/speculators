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
        --output-path ./vocab_mapping.pt
"""

import argparse
import logging
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from speculators.data_generation.vocab_mapping import (
    build_vocab_mappings_from_distribution,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser: ArgumentParser = argparse.ArgumentParser(
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
        required=True,
        help="Vocabulary size for the target model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the vocabulary mapping file (.pt)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.token_freq_path):
        raise FileNotFoundError(
            f"Token frequency file not found: {args.token_freq_path}"
        )

    token_freq_dict = torch.load(args.token_freq_path, weights_only=False)

    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=args.target_vocab_size,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save as .npy files (expected by training script)
    base_dir = output_dir if output_dir else "."
    d2t_path = os.path.join(base_dir, "d2t.npy")
    t2d_path = os.path.join(base_dir, "t2d.npy")

    np.save(d2t_path, d2t.cpu().numpy())
    np.save(t2d_path, t2d.cpu().numpy())

    logger.info(f"Saved d2t to {d2t_path} (shape: {d2t.shape})")
    logger.info(f"Saved t2d to {t2d_path} (shape: {t2d.shape})")


if __name__ == "__main__":
    main()
