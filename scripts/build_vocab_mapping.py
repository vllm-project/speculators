#!/usr/bin/env python3
"""
Build vocabulary mappings (d2t and t2d) from token frequency distribution.

This script takes a token frequency distribution file (generated during data preprocessing)
and creates vocabulary mappings for a draft model with a smaller vocabulary.

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

import torch

from speculators.data_generation.vocab_mapping import build_vocab_mappings_from_distribution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build vocabulary mappings from token frequency distribution"
    )

    parser.add_argument(
        '--token-freq-path',
        type=str,
        required=True,
        help='Path to token frequency distribution file (.pt)'
    )
    parser.add_argument(
        '--draft-vocab-size',
        type=int,
        required=True,
        help='Vocabulary size for the draft model'
    )
    parser.add_argument(
        '--target-vocab-size',
        type=int,
        required=True,
        help='Vocabulary size for the target model'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save the vocabulary mapping file (.pt)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.token_freq_path):
        raise FileNotFoundError(f"Token frequency file not found: {args.token_freq_path}")

    token_freq_dict = torch.load(args.token_freq_path, weights_only=False)

    d2t, t2d = build_vocab_mappings_from_distribution(
        token_freq_dict=token_freq_dict,
        draft_vocab_size=args.draft_vocab_size,
        target_vocab_size=args.target_vocab_size,
    )

    vocab_mapping = {"d2t": d2t, "t2d": t2d}

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(vocab_mapping, args.output_path)
    logger.info(f"Saved vocabulary mapping to {args.output_path} (d2t: {d2t.shape}, t2d: {t2d.shape})")


if __name__ == '__main__':
    main()
