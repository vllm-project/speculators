"""Vocabulary mapping utilities for draft model training."""

import os
from collections import Counter

import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm  # type: ignore[import-untyped]

__all__ = [
    "build_vocab_mappings_from_distribution",
    "save_token_frequency_distribution",
]


def save_token_frequency_distribution(
    dataset: HFDataset,
    output_path: str = "./token_freq.pt",
) -> str:
    """Save token frequency distribution from the dataset.

    Args:
        dataset: HuggingFace dataset with input_ids and loss_mask
        output_path: Path where to save the token frequency distribution

    Returns:
        Path to the saved frequency distribution file
    """
    if os.path.exists(output_path):
        return output_path

    token_freq: Counter[int] = Counter()
    for item in tqdm(dataset, desc="Counting token frequencies"):
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        batch_token_freq = dict(zip(unique_ids.tolist(), counts.tolist(), strict=False))
        token_freq.update(batch_token_freq)

    token_freq_dict = dict(token_freq)
    torch.save(token_freq_dict, output_path)

    return output_path


def build_vocab_mappings_from_distribution(
    token_freq_dict: dict[int, int],
    draft_vocab_size: int,
    target_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build vocabulary mappings for draft model from token frequency distribution."""
    sorted_tokens = sorted(token_freq_dict.items(), key=lambda x: (-x[1], x[0]))

    num_tokens_to_select = min(draft_vocab_size, len(sorted_tokens))
    selected_token_ids = [
        token_id for token_id, _ in sorted_tokens[:num_tokens_to_select]
    ]

    if len(selected_token_ids) < draft_vocab_size:
        current_ids = set(selected_token_ids)
        for tid in range(draft_vocab_size):
            if tid not in current_ids:
                selected_token_ids.append(tid)
            if len(selected_token_ids) >= draft_vocab_size:
                break

    selected_token_ids.sort()

    # Store offset: target_token_id = draft_idx + draft_to_target[draft_idx]
    draft_to_target = torch.zeros(draft_vocab_size, dtype=torch.long)
    for draft_idx, target_token_id in enumerate(selected_token_ids):
        draft_to_target[draft_idx] = target_token_id - draft_idx

    target_to_draft = torch.zeros(target_vocab_size, dtype=torch.bool)
    for target_token_id in selected_token_ids:
        if target_token_id < target_vocab_size:
            target_to_draft[target_token_id] = True

    return draft_to_target, target_to_draft
