import bisect
import random
import re

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


def _visualize_sample(_dataset, preprocessed, tokenizer, idx: int = 0):
    """Visualize a single sample with color-coded trainable regions."""
    # Get preprocessed sample
    prep_sample = preprocessed[idx]
    input_ids = prep_sample["input_ids"]
    loss_mask = prep_sample["loss_mask"]

    log.info(f"SAMPLE #{idx}")
    log.info("HIGHLIGHTED TEXT (BLUE = trainable, GREY = masked)")

    # Create color-highlighted text
    blue = "\033[38;5;153m"  # Very light blue text for trainable tokens
    grey = "\033[90m"  # Grey text for masked tokens
    reset = "\033[0m"  # Reset color

    output = []
    prev_state = None

    for i in range(len(input_ids)):
        is_train = loss_mask[i].item() == 1
        token = tokenizer.decode([input_ids[i].item()])

        # Switch colors when state changes
        if is_train != prev_state:
            output.append(blue if is_train else grey)
            prev_state = is_train

        output.append(token)

    output.append(reset)
    highlighted = "".join(output)

    log.info(highlighted)


def _normalize_conversation(
    conv: list[dict],
    turn_dropout: bool = False,
) -> list[dict]:
    """Normalize conversation to standard format with role/content keys.

    Args:
        conv: Raw conversation turns
        turn_dropout: If True, randomly keeps first N consecutive turns (1 to len(conv))

    Returns:
        Normalized conversation with optional turn dropout applied
    """
    # Randomly pick how many consecutive turns to keep from the start
    num_turns_to_keep = random.randint(1, len(conv)) if turn_dropout else len(conv)

    normalized = []
    for i, turn in enumerate(conv):
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

        # Stop if we've reached the truncation point
        if i + 1 >= num_turns_to_keep and role == "assistant":
            # Only break after an assistant turn
            break

    return normalized


def _supports_assistant_mask(tokenizer: PreTrainedTokenizer) -> bool:
    """Check if tokenizer supports HF assistant token mask."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "test"}],
            tokenize=True,
            return_assistant_tokens_mask=True,
        )
        return True
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def _detect_assistant_pattern(tokenizer: PreTrainedTokenizer) -> str:
    """Auto-detect the assistant message pattern from the tokenizer's chat template.

    Uses multi-turn conversation but extracts pattern from the LAST assistant
    message only.
    """
    test_conv = [
        {"role": "user", "content": "USER_MSG_1"},
        {"role": "assistant", "content": "ASSISTANT_MSG_1"},
        {"role": "user", "content": "USER_MSG_2"},
        {"role": "assistant", "content": "ASSISTANT_MSG_2"},
    ]

    formatted = tokenizer.apply_chat_template(
        test_conv, tokenize=False, add_generation_prompt=False
    )
    assert isinstance(formatted, str), "Expected string from apply_chat_template"

    # Find the LAST assistant message
    second_start = formatted.find("ASSISTANT_MSG_2")
    second_end = second_start + len("ASSISTANT_MSG_2")

    if second_start == -1:
        raise ValueError("Could not detect second assistant message in chat template")

    # Extract role marker from before the last assistant message
    second_user_end = formatted.find("USER_MSG_2") + len("USER_MSG_2")
    prefix = formatted[second_user_end:second_start]

    # Find where the assistant role marker starts
    assistant_pos = prefix.rfind("assistant")
    if assistant_pos != -1:
        role_start = prefix.rfind("<", 0, assistant_pos)
        if role_start != -1:
            role_marker = prefix[role_start:]
        else:
            role_marker = prefix[assistant_pos:]
    else:
        role_marker = prefix

    # Extract suffix from the last assistant message
    suffix = formatted[second_end:]

    # Extract dynamic boundary marker from role_marker
    boundary_match = re.search(
        r"((<\|?[a-zA-Z0-9_]+[\|>]?)|(\[[a-zA-Z0-9_]+\]))", role_marker
    )
    if boundary_match:
        boundary = re.escape(boundary_match.group(1))
        lookahead_pattern = f"(?!{boundary})"
    else:
        # Fallback to hardcoded if no clear tag found
        lookahead_pattern = r"(?!<\|start\|)"

    return (
        re.escape(role_marker)
        + r"((?:"
        + lookahead_pattern
        + r".)*?)"
        + re.escape(suffix)
    )


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

        # Use group(1) to get only the assistant message content,
        # excluding prefix/suffix markers
        span_start_char = match.start(1)
        span_end_char = match.end(1)

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
    turn_dropout: bool = False,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    results: dict[str, list] = {"input_ids": [], "loss_mask": []}
    conversations = examples.get("conversations", [])

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    # Check for mask support once per batch
    supports_mask = _supports_assistant_mask(tokenizer)

    for idx, conv in enumerate(conversations):
        if not conv or not isinstance(conv, list):
            continue

        # Normalize to standard format with optional turn dropout
        normalized_conv = _normalize_conversation(conv, turn_dropout)
        if not normalized_conv:
            continue

        try:
            if supports_mask:
                # HF assistant token mask
                encoded = tokenizer.apply_chat_template(
                    normalized_conv,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=True,
                )

                # input IDs and loss mask
                input_ids = encoded["input_ids"]
                loss_mask = torch.tensor(encoded["assistant_mask"], dtype=torch.long)

            else:
                # Fallback: regex-based detection
                formatted_raw = tokenizer.apply_chat_template(
                    normalized_conv,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                assert isinstance(formatted_raw, str)

                # Tokenize and get offsets
                encoding = tokenizer(
                    formatted_raw,
                    return_offsets_mapping=True,
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False,
                )

                # input IDs and loss mask
                input_ids = encoding["input_ids"]
                offsets = encoding["offset_mapping"]

                loss_mask = _create_loss_mask_from_offsets(
                    formatted_raw, offsets, assistant_pattern
                )

            # Assert shapes match
            assert len(input_ids) == len(loss_mask), (
                f"Shape mismatch: input_ids={len(input_ids)}, "
                f"loss_mask={len(loss_mask)}"
            )

            # Append to results
            results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            results["loss_mask"].append(loss_mask)

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            log.warning(f"Failed to process conversation {idx}: {e}")
            continue

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    num_proc: int = 8,
    assistant_pattern: str | None = None,
    turn_dropout: bool = False,
) -> HFDataset:
    """Build EAGLE3 dataset by tokenizing conversations and creating loss masks.

    Uses the tokenizer's built-in chat template via apply_chat_template.

    Args:
        dataset: Raw dataset with conversations
        tokenizer: Tokenizer with chat template support
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel processing
        assistant_pattern: Optional custom regex pattern for matching assistant
                          responses. If None, pattern will be auto-detected from
                          chat template.
        turn_dropout: If True, randomly keeps first N consecutive turns per
                     conversation
    """
    # Detect and use provided assistant message pattern
    supports_mask = _supports_assistant_mask(tokenizer)

    if assistant_pattern is None and not supports_mask:
        assistant_pattern = _detect_assistant_pattern(tokenizer)
        log.info(f"Detected assistant pattern: {assistant_pattern[:80]}...")
    elif supports_mask:
        log.info("Using HF assistant token mask for loss masking")
    else:
        log.info(f"Using custom assistant pattern: {assistant_pattern[:80]}...")

    original_cols = dataset.column_names

    dataset = dataset.map(
        lambda examples: _preprocess_batch(
            examples, tokenizer, max_length, assistant_pattern, turn_dropout
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
    assistant_pattern: str | None = None,
    turn_dropout: bool = False,
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
        assistant_pattern: Optional custom regex pattern for matching assistant
                          responses. If None, pattern will be auto-detected from
                          chat template.
        turn_dropout: If True, randomly keeps first N consecutive turns per
                     conversation

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
    if turn_dropout:
        log.info("Turn dropout enabled: randomly keeping N consecutive turns")

    preprocessed_dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=seq_length,
        num_proc=build_dataset_num_proc,
        assistant_pattern=assistant_pattern,
        turn_dropout=turn_dropout,
    )

    log.subsection("Computing token frequency distribution")
    save_token_frequency_distribution(
        dataset=preprocessed_dataset,
        output_path=token_freq_path,
    )

    log.subsection("Visualizing sample")
    _visualize_sample(raw_dataset, preprocessed_dataset, tokenizer, idx=0)

    log.section("Dataset preprocessing complete")

    return preprocessed_dataset, tokenizer
