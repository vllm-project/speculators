"""Vocabulary mapping utilities for draft model training."""

from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm  # type: ignore[import-untyped]
from transformers import AutoConfig

__all__ = [
    "build_vocab_mappings_from_distribution",
    "combine_token_frequency_distributions",
    "get_target_vocab_size",
    "save_token_frequency_distribution",
]


def save_token_frequency_distribution(
    dataset: HFDataset,
    output_path: Path | str = "./token_freq.pt",
) -> None:
    """Save token frequency distribution from the dataset.

    Only tokens where ``loss_mask`` is 1 (assistant tokens) are counted. If
    ``output_path`` already exists, the dataset is skipped and the existing
    file is left untouched.

    Args:
        dataset: HuggingFace dataset with input_ids and loss_mask
        output_path: Path where to save the token frequency distribution

    Returns:
        None. The frequency distribution is written to ``output_path``.
    """
    path = Path(output_path)
    if path.exists():
        return

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


def combine_token_frequency_distributions(
    token_freq_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """Combine multiple token frequency distributions into a single file.

    Args:
        token_freq_paths: Paths of token frequency files, as written by
            :func:`save_token_frequency_distribution`.
        output_path: Path where to save the combined frequency distribution.

    Returns:
        None. The combined frequency distribution is written to
        ``output_path``.
    """
    token_freq_dicts: list[dict[int, int]] = [
        torch.load(path, weights_only=True) for path in token_freq_paths
    ]
    combined_token_freq: Counter[int] = Counter()
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


def get_target_vocab_size(
    target_vocab_size: int | None,
    target_model_path: str | Path | None,
    trust_remote_code: bool = False,
) -> int:
    """Resolve the vocabulary size of the target (verifier) model.

    Exactly one of ``target_vocab_size`` and ``target_model_path`` must be
    provided. When a model path is given, the vocabulary size is read from
    the model config, unwrapping ``text_config`` for multimodal models.

    Args:
        target_vocab_size: Explicit vocabulary size of the target model.
        target_model_path: Path or model name of the target model to load
            the config from.
        trust_remote_code: Whether to trust remote code when loading the
            model config.

    Returns:
        The target model's vocabulary size.

    Raises:
        ValueError: If both or neither of ``target_vocab_size`` and
            ``target_model_path`` are provided.
    """
    if target_vocab_size is not None and target_model_path is not None:
        raise ValueError("Cannot specify both target-vocab-size and target-model-path")

    if target_vocab_size is not None:
        return target_vocab_size

    if target_model_path is None:
        raise ValueError("Must specify either target-vocab-size or target-model-path")

    config = AutoConfig.from_pretrained(
        target_model_path,
        trust_remote_code=trust_remote_code,
    )

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(config, "text_config"):
        config = config.text_config

    return config.vocab_size
