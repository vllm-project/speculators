"""
Vocabulary mapping utilities for draft model training.

This module provides functionality to:
- Count token frequencies from preprocessed datasets
- Build vocabulary mappings from target model to draft model
"""

import os
from collections import Counter

import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm


def save_token_frequency_distribution(
    dataset: HFDataset,
    cache_dir: str = "./cache/token_frequencies",
    cache_key: str = "token_freq",
) -> str:
    """
    Save token frequency distribution from the dataset.

    This counts how often each token appears in the trainable portions
    (where loss_mask=1) of the dataset and saves it for later use.

    Args:
        dataset: The processed dataset with input_ids and loss_mask
        cache_dir: Directory for saving the frequency distribution
        cache_key: Key for the cache file

    Returns:
        Path to the saved frequency distribution file
    """
    os.makedirs(cache_dir, exist_ok=True)
    freq_dist_path = os.path.join(cache_dir, f"{cache_key}_token_freq.pt")

    if os.path.exists(freq_dist_path):
        return freq_dist_path

    token_freq: Counter[int] = Counter()
    for item in tqdm(dataset, desc="Counting token frequencies"):
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        batch_token_freq = dict(zip(unique_ids.tolist(), counts.tolist(), strict=False))
        token_freq.update(batch_token_freq)

    token_freq_dict = dict(token_freq)
    torch.save(token_freq_dict, freq_dist_path)

    return freq_dist_path


def build_vocab_mappings_from_distribution(
    token_freq_dict: dict[int, int],
    draft_vocab_size: int,
    target_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build vocabulary mappings for draft model from token frequency distribution.

    Creates two mappings:
    1. draft_to_target: Offset array where draft_to_target[i] + i gives target token ID
    2. target_to_draft: Boolean array indicating which target tokens are in draft vocab

    Args:
        token_freq_dict: Dictionary mapping token IDs to their frequencies
        draft_vocab_size: Size of the draft model vocabulary
        target_vocab_size: Size of the target model vocabulary

    Returns:
        Tuple of (draft_to_target, target_to_draft) tensors:
        - draft_to_target: Offset mapping from draft to target IDs
          (shape: [draft_vocab_size])
        - target_to_draft: Boolean mask indicating target tokens in draft vocab
          (shape: [target_vocab_size])
    """
    # Sort tokens by frequency (descending) to get most common tokens
    sorted_tokens = sorted(token_freq_dict.items(), key=lambda x: (-x[1], x[0]))

    # Take top N most frequent tokens for draft vocabulary
    num_tokens_to_select = min(draft_vocab_size, len(sorted_tokens))
    selected_token_ids = [
        token_id for token_id, _ in sorted_tokens[:num_tokens_to_select]
    ]

    # Fill remaining slots with low token IDs if we don't have enough
    if len(selected_token_ids) < draft_vocab_size:
        current_ids = set(selected_token_ids)
        for tid in range(draft_vocab_size):
            if tid not in current_ids:
                selected_token_ids.append(tid)
            if len(selected_token_ids) >= draft_vocab_size:
                break

    # Sort selected tokens by ID for consistent ordering
    selected_token_ids.sort()

    # Build draft_to_target: stores the offset needed to map draft index
    # to target token ID
    # Formula: target_token_id = draft_index + draft_to_target[draft_index]
    draft_to_target = torch.zeros(draft_vocab_size, dtype=torch.long)
    for draft_idx, target_token_id in enumerate(selected_token_ids):
        draft_to_target[draft_idx] = target_token_id - draft_idx

    # Build target_to_draft: boolean array indicating presence in draft vocab
    target_to_draft = torch.zeros(target_vocab_size, dtype=torch.bool)
    for target_token_id in selected_token_ids:
        if target_token_id < target_vocab_size:
            target_to_draft[target_token_id] = True

    return draft_to_target, target_to_draft
