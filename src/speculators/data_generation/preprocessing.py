import bisect
import re

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


def _normalize_conversation(conv: list[dict]) -> list[dict]:
    """Normalize conversation to standard format with role/content keys."""
    normalized = []
    for turn in conv:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))

        # Map various role names to standard user/assistant
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            log.warning(f"Unknown role '{role}', skipping turn")
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def _detect_assistant_pattern(tokenizer: PreTrainedTokenizer) -> str:
    """Auto-detect the assistant message pattern from the tokenizer's
    chat template using a dummy example.

    TODO: Replace this with return_assistant_tokens_mask when more models
    support it natively.
    """

    test_conv = [
        {"role": "user", "content": "USER_MSG"},
        {"role": "assistant", "content": "ASSISTANT_MSG"},
    ]

    formatted_raw = tokenizer.apply_chat_template(
        test_conv, tokenize=False, add_generation_prompt=False
    )
    # Type assertion: when tokenize=False, result is always str
    assert isinstance(formatted_raw, str)
    formatted: str = formatted_raw

    assistant_start = formatted.find("ASSISTANT_MSG")
    if assistant_start == -1:
        raise ValueError("Could not detect assistant message in chat template")

    assistant_end = assistant_start + len("ASSISTANT_MSG")
    user_end = formatted.find("USER_MSG") + len("USER_MSG")
    prefix = formatted[user_end:assistant_start]

    suffix = formatted[assistant_end:]
    return re.escape(prefix) + r"(.*?)" + re.escape(suffix)


def _create_loss_mask_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    assistant_pattern: str,
) -> torch.Tensor:
    """Create loss mask by finding assistant response spans in formatted text."""
    loss_mask = torch.zeros(len(offsets), dtype=torch.long)

    matches_found = 0
    token_starts = [offset[0] for offset in offsets]

    for match in re.finditer(assistant_pattern, text, re.DOTALL):
        matches_found += 1

        span_start_char = match.start()
        span_end_char = match.end()
        start_idx = bisect.bisect_left(token_starts, span_start_char)

        for idx in range(max(0, start_idx - 1), len(offsets)):
            token_start, token_end = offsets[idx]
            if token_start >= span_end_char:
                break
            # Mark token as trainable if it overlaps with assistant span
            if token_end > span_start_char and token_start < span_end_char:
                loss_mask[idx] = 1

    if matches_found == 0:
        log.warning("No assistant response spans found in conversation")

    return loss_mask


def _preprocess_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    assistant_pattern: str,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    results: dict[str, list] = {"input_ids": [], "loss_mask": []}
    conversations = examples.get("conversations", [])

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    for idx, conv in enumerate(conversations):
        if not conv or not isinstance(conv, list):
            continue

        # Normalize to standard format
        normalized_conv = _normalize_conversation(conv)
        if not normalized_conv:
            continue

        try:
            # Get formatted text with chat template
            formatted_raw = tokenizer.apply_chat_template(
                normalized_conv,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize with offsets
            encoding = tokenizer(
                formatted_raw,
                return_offsets_mapping=True,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )

            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            # Create loss mask using character offsets
            loss_mask = _create_loss_mask_from_offsets(
                formatted_raw, offsets, assistant_pattern
            )

            # Verify shapes match exactly
            assert len(input_ids) == len(loss_mask), (
                f"Shape mismatch: input_ids={len(input_ids)}, "
                f"loss_mask={len(loss_mask)}"
            )

            results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            results["loss_mask"].append(loss_mask)

        except Exception as e:
            log.warning(f"Failed to process conversation {idx}: {e}")
            continue

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    num_proc: int = 8,
) -> HFDataset:
    """Build EAGLE3 dataset by tokenizing conversations and creating loss masks.

    Uses the tokenizer's built-in chat template via apply_chat_template.
    """
    # Detect assistant message pattern from chat template
    assistant_pattern = _detect_assistant_pattern(tokenizer)
    log.info(f"Detected assistant pattern: {assistant_pattern[:80]}...")

    original_cols = dataset.column_names

    dataset = dataset.map(
        lambda examples: _preprocess_batch(
            examples, tokenizer, max_length, assistant_pattern
        ),
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        load_from_cache_file=True,
    )

    dataset.set_format(type="torch")
    return dataset


def load_raw_dataset(
    train_data_path: str, num_proc: int = 8, cache_dir: str | None = None
) -> HFDataset:
    """Load raw dataset from local file or HuggingFace."""
    if train_data_path.endswith((".jsonl", ".json")):
        return load_dataset(
            "json", data_files=train_data_path, split="train", cache_dir=cache_dir
        )

    if train_data_path not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {train_data_path}. "
            f"Supported: local .json/.jsonl files or {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[train_data_path]
    raw_dataset = load_dataset(config.hf_path, split=config.split, cache_dir=cache_dir)

    if config.normalize_fn is not None:
        raw_dataset = raw_dataset.map(config.normalize_fn, num_proc=num_proc)

    return raw_dataset


def load_and_preprocess_dataset(
    target_model_path: str,
    train_data_path: str,
    seq_length: int,
    build_dataset_num_proc: int = 8,
    seed: int = 0,
    max_samples: int | None = None,
    token_freq_path: str = "./token_freq.pt",  # noqa: S107
    cache_dir: str | None = None,
) -> tuple[HFDataset, PreTrainedTokenizer]:
    """Load, tokenize, and preprocess a dataset for EAGLE3 training.

    Uses the tokenizer's built-in chat template via apply_chat_template.
    Caching is handled automatically by HuggingFace datasets.

    Args:
        target_model_path: HuggingFace model ID or local path
        train_data_path: Dataset name or path to JSON/JSONL file
        seq_length: Maximum sequence length
        build_dataset_num_proc: Number of processes for dataset building
        seed: Random seed for shuffling
        max_samples: Optional limit on number of samples
        token_freq_path: Path to save token frequency distribution
        cache_dir: Directory to cache HuggingFace datasets (optional)

    Returns:
        Tuple of (preprocessed_dataset, tokenizer)
    """
    log.section("Starting dataset preprocessing")

    log.subsection("Loading tokenizer and dataset")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {target_model_path} does not support chat templates. "
            "Please use a model with a pre-configured chat template."
        )

    raw_dataset = load_raw_dataset(
        train_data_path, num_proc=build_dataset_num_proc, cache_dir=cache_dir
    )
    raw_dataset = raw_dataset.shuffle(seed=seed)

    if max_samples is not None and len(raw_dataset) > max_samples:
        raw_dataset = raw_dataset.select(range(max_samples))

    log.info(f"Loaded {len(raw_dataset)} samples")

    log.subsection("Tokenizing and building dataset")
    if cache_dir:
        log.info(f"Preprocessed data will be cached at: {cache_dir}")

    preprocessed_dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=seq_length,
        num_proc=build_dataset_num_proc,
    )

    log.subsection("Computing token frequency distribution")
    save_token_frequency_distribution(
        dataset=preprocessed_dataset,
        output_path=token_freq_path,
    )

    log.section("Dataset preprocessing complete")

    return preprocessed_dataset, tokenizer
