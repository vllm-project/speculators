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

__all__ = [
    "build_eagle3_dataset",
    "generate_cache_key",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
    "view_samples",
]

log = PipelineLogger(__name__)


def _apply_loss_mask_from_chat_template(
    text: str,
    offsets: torch.Tensor,
    chat_template: ChatTemplate,
) -> torch.Tensor:
    """Apply loss mask to identify assistant response spans."""
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
        log.warning("No assistant response spans found in the conversation text.")

    return loss_mask


def _preprocess_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    template: ChatTemplate,
    max_length: int,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""
    results: dict[str, list] = {"input_ids": [], "loss_mask": []}

    conversations = examples.get("conversations", [])
    if not conversations:
        return results

    for conv in conversations:
        if not conv or not isinstance(conv, list):
            continue

        text = format_conversation(conv, template)

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
    """Build EAGLE3 dataset by tokenizing conversations and creating loss masks."""
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(
            f"Chat template '{chat_template}' not found. "
            f"Available templates: {list(CHAT_TEMPLATES.keys())}"
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
    """Load raw dataset from local file or HuggingFace."""
    if train_data_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=train_data_path, split="train")

    if train_data_path not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {train_data_path}. "
            f"Supported: local .json/.jsonl files or {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[train_data_path]
    raw_dataset = load_dataset(config.hf_path, split=config.split)

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
    """Load, tokenize, and preprocess a dataset for EAGLE3 training."""
    log.section("Starting dataset preprocessing")

    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(
            f"Chat template '{chat_template}' not found. "
            f"Available: {list(CHAT_TEMPLATES.keys())}"
        )

    log.subsection("Loading tokenizer and dataset")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_raw_dataset(train_data_path, num_proc=build_dataset_num_proc)
    raw_dataset = raw_dataset.shuffle(seed=seed)

    if max_samples is not None and len(raw_dataset) > max_samples:
        raw_dataset = raw_dataset.select(range(max_samples))

    log.info(f"Loaded {len(raw_dataset)} samples")

    cache_key = generate_cache_key(
        target_model_path, chat_template, seq_length, train_data_path
    )
    if max_samples is not None:
        cache_key = f"{cache_key}_samples{max_samples}"
    dataset_cache_dir = os.path.join(cache_dir, "processed_dataset", cache_key)
    os.makedirs(dataset_cache_dir, exist_ok=True)

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

        preprocessed_dataset.save_to_disk(dataset_cache_dir)
        log.info(f"Saved preprocessed dataset to {dataset_cache_dir}")

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
