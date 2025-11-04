import bisect
import hashlib
import os
import random
import re

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer

from .configs import CHAT_TEMPLATES, DATASET_CONFIGS, ChatTemplate, format_conversation
from .logging_utils import PipelineLogger
from .vocab_mapping import save_token_frequency_distribution

log = PipelineLogger(__name__)


def _apply_loss_mask_from_chat_template(
    text: str,
    offsets: torch.Tensor,
    chat_template: ChatTemplate,
) -> torch.Tensor:
    """
    Apply loss mask to identify assistant response spans using chat template.

    Args:
        text: The formatted conversation text
        offsets: Token offset mapping from tokenizer
        chat_template: The chat template to use for identifying assistant spans

    Returns:
        A tensor indicating which tokens should contribute to the loss (1) or not (0)
    """
    loss_mask = torch.zeros(len(offsets), dtype=torch.long)

    user_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.user_header}"
    )
    assistant_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
    )

    assistant_pattern = (
        re.escape(assistant_message_separator)
        + r"(.*?)(?="
        + re.escape(user_message_separator)
        + "|$)"
    )

    matches_found = 0
    token_starts = [int(offset[0]) for offset in offsets]

    for match in re.finditer(assistant_pattern, text, re.DOTALL):
        matches_found += 1
        assistant_start_char = match.start(1)
        assistant_end_char = match.end(1)

        start_idx = bisect.bisect_left(token_starts, assistant_start_char)

        for idx in range(max(0, start_idx - 1), len(offsets)):
            token_start, token_end = int(offsets[idx][0]), int(offsets[idx][1])
            if token_start > assistant_end_char:
                break
            if token_end > assistant_start_char and token_start <= assistant_end_char:
                loss_mask[idx] = 1

    if matches_found == 0:
        log.warning(
            "No assistant response spans found in conversation. "
            "Verify chat template matches your data format and that "
            "conversations contain assistant responses."
        )

    return loss_mask


def _preprocess_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    template: ChatTemplate,
    max_length: int,
) -> dict[str, list]:
    """
    Process a batch of conversations into tokenized format with loss masks.

    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer to use
        template: Chat template for formatting
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids and loss_mask lists
    """
    results: dict[str, list] = {"input_ids": [], "loss_mask": []}

    conversations = examples.get("conversations", [])
    if not conversations:
        return results

    for conv in conversations:
        if not conv or not isinstance(conv, list):
            continue

        text = format_conversation(conv, template)

        # Tokenize with offset mapping for loss mask
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoded.input_ids[0]
        offsets = encoded.offset_mapping[0]

        loss_mask = _apply_loss_mask_from_chat_template(text, offsets, template)

        results["input_ids"].append(input_ids)
        results["loss_mask"].append(loss_mask)

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    max_length: int = 2048,
    num_proc: int = 8,
) -> HFDataset:
    """
    Build EAGLE3 dataset by tokenizing conversations and creating loss masks.

    Args:
        dataset: HF dataset to process with "conversations" column in ShareGPT format
        tokenizer: The tokenizer to use for tokenization
        chat_template: The chat template identifier (e.g., "qwen2", "llama3")
        max_length: Maximum sequence length
        num_proc: Number of processes for multiprocessing

    Returns:
        Processed HF dataset with input_ids and loss_mask
    """
    # Get the chat template
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(
            f"Chat template '{chat_template}' not found. "
            f"Available templates: {', '.join(sorted(CHAT_TEMPLATES.keys()))}"
        )
    template = CHAT_TEMPLATES[chat_template]
    original_cols = dataset.column_names

    dataset = dataset.map(
        lambda examples: _preprocess_batch(examples, tokenizer, template, max_length),
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        load_from_cache_file=False,
    )

    dataset.set_format(type="torch")
    return dataset


def load_raw_dataset(train_data_path: str, num_proc: int = 8) -> HFDataset:
    """
    Load raw dataset from local file or HuggingFace.

    Supports:
    - Local .json/.jsonl files
    - HuggingFace dataset shortcuts (see DATASET_CONFIGS)

    Args:
        train_data_path: Path to local file or dataset name
        num_proc: Number of processes for preprocessing

    Returns:
        HuggingFace Dataset with conversations in standard format
    """
    # Load from local file
    if train_data_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=train_data_path, split="train")

    # Load from HuggingFace using registry
    if train_data_path not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: '{train_data_path}'. "
            f"Supported options:\n"
            f"  - Local files: .json or .jsonl files\n"
            f"  - Registered datasets: {', '.join(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[train_data_path]
    raw_dataset = load_dataset(config.hf_path, split=config.split)

    # Apply normalization if configured
    if config.normalize_fn is not None:
        raw_dataset = raw_dataset.map(config.normalize_fn, num_proc=num_proc)

    return raw_dataset


def generate_cache_key(
    target_model_path: str,
    chat_template: str,
    seq_length: int,
    train_data_path: str,
) -> str:
    """Generate MD5 cache key from preprocessing parameters."""
    key_string = f"{target_model_path}_{chat_template}_{seq_length}_{train_data_path}"
    return hashlib.md5(key_string.encode()).hexdigest()


def load_and_preprocess_dataset(
    target_model_path: str,
    train_data_path: str,
    chat_template: str,
    seq_length: int,
    cache_dir: str,
    build_dataset_num_proc: int = 8,
    seed: int = 0,
    max_samples: int | None = None,
) -> tuple[HFDataset, PreTrainedTokenizer]:
    """
    Load, tokenize, and preprocess a dataset for EAGLE3 training.

    Args:
        target_model_path: HuggingFace model ID or local path
        train_data_path: Path to training data (JSON/JSONL) or dataset name
            (sharegpt/ultrachat)
        chat_template: Chat template identifier
        seq_length: Maximum sequence length
        cache_dir: Directory for caching
        build_dataset_num_proc: Number of processes for dataset building
        seed: Random seed for shuffling
        max_samples: Maximum number of samples to process (None = process all)

    Returns:
        Tuple of (preprocessed_dataset, tokenizer)
    """
    log.section("Starting dataset preprocessing")

    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(
            f"Chat template '{chat_template}' not found. "
            f"Available: {', '.join(sorted(CHAT_TEMPLATES.keys()))}"
        )

    log.subsection("Loading tokenizer and dataset")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_raw_dataset(train_data_path, num_proc=build_dataset_num_proc)
    raw_dataset = raw_dataset.shuffle(seed=seed)

    # Limit dataset size if requested
    if max_samples is not None and len(raw_dataset) > max_samples:
        raw_dataset = raw_dataset.select(range(max_samples))

    log.info(f"Loaded {len(raw_dataset)} samples")

    # Prepare cache directories
    cache_key = generate_cache_key(
        target_model_path, chat_template, seq_length, train_data_path
    )
    if max_samples is not None:
        cache_key = f"{cache_key}_samples{max_samples}"
    dataset_cache_dir = os.path.join(cache_dir, "processed_dataset", cache_key)
    os.makedirs(dataset_cache_dir, exist_ok=True)

    # Check if already cached
    if os.path.exists(os.path.join(dataset_cache_dir, "dataset_info.json")):
        log.info(f"Loading cached dataset from {dataset_cache_dir}")
        preprocessed_dataset = load_from_disk(dataset_cache_dir)
    else:
        log.subsection("Tokenizing and building dataset")
        preprocessed_dataset = build_eagle3_dataset(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=seq_length,
            num_proc=build_dataset_num_proc,
        )

        # Save to disk
        preprocessed_dataset.save_to_disk(dataset_cache_dir)
        log.info(f"Saved preprocessed dataset to {dataset_cache_dir}")

    # Save token frequency distribution (for later vocab mapping generation)
    log.subsection("Computing token frequency distribution")
    token_freq_cache_dir = os.path.join(cache_dir, "token_frequencies")
    save_token_frequency_distribution(
        dataset=preprocessed_dataset,
        cache_dir=token_freq_cache_dir,
        cache_key=cache_key,
    )

    log.section("Dataset preprocessing complete")

    return preprocessed_dataset, tokenizer


def view_samples(
    dataset: HFDataset, tokenizer: PreTrainedTokenizer, num_samples: int = 3
):
    """View random samples for sanity check."""
    log.section(f"Viewing {num_samples} random samples")

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        input_ids = sample["input_ids"].squeeze()
        loss_mask = sample["loss_mask"].squeeze()

        log.info(f"\nSample {idx}:")
        log.info(
            f"  Shape: {input_ids.shape}, Trainable tokens: {loss_mask.sum().item()}"
        )
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        log.info(f"  Text (first 500 chars):\n{decoded_text[:500]}...")

        trainable_ids = input_ids[loss_mask == 1]
        if len(trainable_ids) > 0:
            trainable_text = tokenizer.decode(trainable_ids, skip_special_tokens=False)
            log.info(f"  Trainable text (first 200 chars):\n{trainable_text[:200]}...")
