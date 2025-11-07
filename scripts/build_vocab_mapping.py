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
from pathlib import Path

import numpy as np
import torch

from speculators.data_generation.vocab_mapping import (
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


def main():
    args = parse_args()

    token_freq_path = Path(args.token_freq_path)
    if not token_freq_path.exists():
        raise FileNotFoundError(f"Token frequency file not found: {token_freq_path}")

    token_freq_dict = torch.load(token_freq_path, weights_only=True)

    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=args.target_vocab_size,
    )

    output_path = Path(args.output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as .npy files (expected by training script)
    d2t_path = output_dir / "d2t.npy"
    t2d_path = output_dir / "t2d.npy"

    np.save(d2t_path, d2t.cpu().numpy())
    np.save(t2d_path, t2d.cpu().numpy())

    logger.info(f"Saved d2t to {d2t_path} (shape: {d2t.shape})")
    logger.info(f"Saved t2d to {t2d_path} (shape: {t2d.shape})")


if __name__ == "__main__":
    main()
