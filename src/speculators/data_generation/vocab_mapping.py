"""Vocabulary mapping utilities for draft model training."""

from collections import Counter
from pathlib import Path

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
    path = Path(output_path)
    if path.exists():
        return output_path

    token_freq: Counter[int] = Counter()
    for item in tqdm(dataset, desc="Counting token frequencies"):
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        # Only count tokens where loss_mask is 1 (assistant tokens)
        masked_token_ids = input_ids[loss_mask.to(torch.bool)]
        unique_ids, counts = masked_token_ids.unique(return_counts=True)
        batch_token_freq = dict(zip(unique_ids.tolist(), counts.tolist(), strict=True))
        token_freq.update(batch_token_freq)

    token_freq_dict = dict(token_freq)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_freq_dict, path)

    return output_path


def combine_token_frequency_distributions(
    token_freq_paths: list[str | Path],
    output_path: str | Path,
):
    """Combine multiple token frequency distributions into a single file."""
    token_freq_dicts = [
        torch.load(path, weights_only=True) for path in token_freq_paths
    ]
    combined_token_freq = Counter()
    for token_freq_dict in token_freq_dicts:
        combined_token_freq.update(token_freq_dict)
    combined_token_freq_dict = dict(combined_token_freq)
    torch.save(combined_token_freq_dict, output_path)


def build_vocab_mappings_from_distribution(
    token_freq_dict: dict[int, int],
    draft_vocab_size: int,
    target_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build vocabulary mappings for draft model from token frequency distribution."""

    sorted_tokens = sorted(
        token_freq_dict, key=lambda tid: (-token_freq_dict[tid], tid)
    )

    num_tokens_to_select = min(draft_vocab_size, len(sorted_tokens))
    selected_token_ids = sorted_tokens[:num_tokens_to_select]

    if len(selected_token_ids) < draft_vocab_size:
        current_ids = set(selected_token_ids)
        for tid in range(draft_vocab_size):
            if tid not in current_ids:
                selected_token_ids.append(tid)
            if len(selected_token_ids) >= draft_vocab_size:
                break

    selected_token_ids.sort()

    # Store offset: target_token_id = draft_idx + draft_to_target[draft_idx]
    draft_to_target = torch.tensor(selected_token_ids, dtype=torch.long) - torch.arange(
        draft_vocab_size, dtype=torch.long
    )

    target_to_draft = torch.zeros(target_vocab_size, dtype=torch.bool)
    target_to_draft[selected_token_ids] = True

    return draft_to_target, target_to_draft
