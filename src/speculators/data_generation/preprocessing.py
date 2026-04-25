import bisect
import random
import re
from functools import partial
from pathlib import Path
from re import Pattern
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


def _visualize_sample(preprocessed, tokenizer, idx: int = 0):
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


def _normalize_content(content: Any) -> Any:
    """Normalize multimodal content into a processor-friendly representation."""

    def _strip_image_tokens(text: str) -> str:
        return re.sub(r"<image\s*/?>", "", text, flags=re.IGNORECASE).strip()

    if isinstance(content, list):
        normalized_items = []
        for item in content:
            if isinstance(item, str):
                normalized_items.append(
                    {"type": "text", "text": _strip_image_tokens(item)}
                )
                continue
            if isinstance(item, dict):
                image_ref = _get_image_ref(item)
                if image_ref is not None:
                    normalized_items.append({"type": "image", "image": image_ref})
                    continue
                text_val = item.get("text")
                if text_val is not None:
                    normalized_items.append(
                        {"type": "text", "text": _strip_image_tokens(text_val)}
                    )
                    continue
            normalized_items.append(item)
        return normalized_items
    return content


def _get_conversations_from_examples(examples: dict) -> list:
    """Extract conversations/messages from a batched dataset example."""
    if "conversations" in examples:
        return examples.get("conversations", [])
    if "messages" in examples:
        return examples.get("messages", [])
    return []


def _flatten_singleton_batch(value: Any, *, field_name: str) -> Any:
    """Collapse a singleton batch dimension from HF outputs."""
    if isinstance(value, torch.Tensor):
        value = value.tolist()

    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(
                f"{field_name} returned non-singleton batch: batch_size={len(value)}"
            )
        return value[0]

    return value


def _get_tokenizer_from_processor(processor: Any) -> PreTrainedTokenizerBase:
    """Extract the tokenizer from a processor."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not provide a tokenizer attribute.")
    return cast("PreTrainedTokenizerBase", tokenizer)


def _get_image_ref(item: dict) -> Any | None:
    """Extract a serializable image reference from a multimodal content item."""
    if item.get("type") not in ("image", "image_url"):
        return None

    image_ref = item.get("image")
    if image_ref is not None:
        return image_ref

    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    return image_url


def _extract_multimodal_data_from_conversation(
    conv: list[dict],
) -> dict[str, list[Any]]:
    """Build the minimal multi_modal_data payload expected by vLLM."""
    mm_data: dict[str, list[Any]] = {}
    for turn in conv:
        content = turn.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue
            image_ref = _get_image_ref(item)
            if image_ref is not None:
                mm_data.setdefault("image", []).append(image_ref)

    return mm_data


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
        content = turn.get("value") or turn.get("content") or ""
        content = _normalize_content(content)

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

        # Build normalized turn with role and content
        normalized_turn = {"role": role, "content": content}

        # Preserve 'thinking' field if it exists
        if "thinking" in turn and turn["thinking"]:
            normalized_turn["thinking"] = turn["thinking"]

        normalized.append(normalized_turn)

        # Stop if we've reached the truncation point
        if i + 1 >= num_turns_to_keep and role == "assistant":
            # Only break after an assistant turn
            break

    return normalized


def _supports_assistant_mask(caller: Any) -> bool:
    """Check if tokenizer/processor truly supports HF assistant token mask.

    Must return a non-zero mask for a conversation containing an assistant message.
    """
    try:
        res_any = caller.apply_chat_template(
            [{"role": "assistant", "content": "test"}],
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        res = cast("dict[str, Any]", res_any)
        # Check both singular and plural key names
        mask = res.get("assistant_masks", res.get("assistant_mask"))
        if mask is None:
            return False

        # Verify the mask is not all zeros
        mask = _flatten_singleton_batch(mask, field_name="assistant mask")
        return any(m == 1 for m in mask)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def _detect_assistant_pattern(caller: Any) -> str:
    """Auto-detect the assistant message pattern from a chat template caller.

    Uses multi-turn conversation but extracts pattern from the LAST assistant
    message only.
    """
    test_conv = [
        {"role": "user", "content": "USER_MSG_1"},
        {"role": "assistant", "content": "ASSISTANT_MSG_1"},
        {"role": "user", "content": "USER_MSG_2"},
        {"role": "assistant", "content": "ASSISTANT_MSG_2"},
    ]

    formatted = caller.apply_chat_template(
        test_conv, tokenize=False, add_generation_prompt=False
    )
    assert isinstance(formatted, str), "Expected string from apply_chat_template"

    # Find the START and END of both assistant messages
    first_start = formatted.find("ASSISTANT_MSG_1")
    first_end = first_start + len("ASSISTANT_MSG_1")
    second_start = formatted.find("ASSISTANT_MSG_2")
    second_end = second_start + len("ASSISTANT_MSG_2")

    if first_start == -1 or second_start == -1:
        raise ValueError("Could not detect assistant messages in chat template")

    # Extract role marker from before the second assistant message
    second_user_end = formatted.find("USER_MSG_2") + len("USER_MSG_2")
    prefix = formatted[second_user_end:second_start]

    # Find where the assistant role marker starts
    assistant_pos = prefix.rfind("assistant")
    if assistant_pos != -1:
        # Search for a tag start ('<' or '[') before 'assistant'
        role_start = -1
        for char in ["<", "["]:
            pos = prefix.rfind(char, 0, assistant_pos)
            role_start = max(role_start, pos)
        if role_start != -1:
            role_marker = prefix[role_start:]
        else:
            role_marker = prefix[assistant_pos:]
    else:
        role_marker = prefix

    # Strip <think>...</think> blocks from the role marker. Thinking model
    # templates wrap assistant content in these tags, but the test messages
    # can produce empty blocks (e.g. "<think>\n\n</think>\n") with reasoning models,
    # which then get baked into the regex as literals. Removing them ensures
    # that reasoning stays within the assistant content group.
    role_marker = re.sub(r"<think>.*?</think>\s*", "", role_marker, flags=re.DOTALL)

    # Determine the stable TURN-LEVEL suffix
    suffix1 = formatted[first_end : formatted.find("USER_MSG_2")]
    suffix2 = formatted[second_end:]

    # The stable suffix is the common prefix of these two tails
    common_len = 0
    for c1, c2 in zip(suffix1, suffix2, strict=False):
        if c1 == c2:
            common_len += 1
        else:
            break
    suffix = suffix1[:common_len]

    if not suffix:
        suffix = suffix1 if suffix1 else "\n"

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
    assistant_pattern: str | Pattern[str],
) -> torch.Tensor:
    """Create loss mask by finding assistant response spans in formatted text."""
    loss_mask = torch.zeros(len(offsets), dtype=torch.bool)

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
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    conversations = _get_conversations_from_examples(examples)

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    for idx, conv in enumerate(conversations):
        if not conv or not isinstance(conv, list):
            continue

        # Normalize to standard format with optional turn dropout
        normalized_conv = _normalize_conversation(conv, turn_dropout)
        if not normalized_conv:
            continue

        try:
            if assistant_pattern is None:
                # HF assistant token mask
                encoded_any = tokenizer.apply_chat_template(
                    normalized_conv,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                )
                encoded = cast("dict[str, Any]", encoded_any)

                # input IDs and loss mask
                input_ids = _flatten_singleton_batch(
                    encoded["input_ids"],
                    field_name="Text apply_chat_template input_ids",
                )
                # HF uses 'assistant_masks' in recent versions
                mask_key = (
                    "assistant_masks"
                    if "assistant_masks" in encoded
                    else "assistant_mask"
                )
                loss_mask = torch.tensor(
                    _flatten_singleton_batch(
                        encoded[mask_key],
                        field_name="Text apply_chat_template assistant mask",
                    ),
                    dtype=torch.long,
                )

            else:
                # Fallback: regex-based detection
                assert assistant_pattern is not None, (
                    "Assistant pattern required for fallback"
                )
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
                input_ids = _flatten_singleton_batch(
                    encoding["input_ids"], field_name="Text tokenizer input_ids"
                )
                offsets = _flatten_singleton_batch(
                    encoding["offset_mapping"],
                    field_name="Text tokenizer offset_mapping",
                )

                loss_mask = _create_loss_mask_from_offsets(
                    formatted_raw, offsets, assistant_pattern
                )

            # Assert shapes match
            assert len(input_ids) == len(loss_mask), (
                f"Shape mismatch: input_ids={len(input_ids)}, "
                f"loss_mask={len(loss_mask)}"
            )

            # Filtering samples out with too few valid tokens
            if minimum_valid_tokens is not None:
                num_valid_tokens = int(loss_mask.sum().item())
                if num_valid_tokens < minimum_valid_tokens:
                    continue

            # Append to results
            results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            results["loss_mask"].append(loss_mask)
            results["seq_len"].append(len(input_ids))

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            log.error(
                f"Failed to process conversation {idx} "
                f"(assistant_pattern={assistant_pattern is not None}): {e}"
            )
            continue

    return results


def _preprocess_batch_multimodal(  # noqa: PLR0912, PLR0915
    examples: dict,
    processor: Any,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
) -> dict[str, list]:
    """Process a batch of multimodal conversations into token IDs and loss masks."""

    results: dict[str, list] = {
        "input_ids": [],
        "loss_mask": [],
        "multi_modal_data": [],
        "prompt": [],
        "seq_len": [],
    }
    conversations = _get_conversations_from_examples(examples)

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    tokenizer = _get_tokenizer_from_processor(processor)

    for idx, conv in enumerate(conversations):
        if not conv or not isinstance(conv, list):
            continue

        normalized_conv = _normalize_conversation(conv, turn_dropout)
        if not normalized_conv:
            continue

        try:
            formatted_raw = processor.apply_chat_template(
                normalized_conv,
                tokenize=False,
                add_generation_prompt=False,
            )
            assert isinstance(formatted_raw, str)

            if assistant_pattern is None:
                encoded_any = processor.apply_chat_template(
                    normalized_conv,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                )
                encoded = cast("dict[str, Any]", encoded_any)

                input_ids = _flatten_singleton_batch(
                    encoded["input_ids"],
                    field_name="Multimodal apply_chat_template input_ids",
                )
                mask_key = (
                    "assistant_masks"
                    if "assistant_masks" in encoded
                    else "assistant_mask"
                )
                loss_mask = torch.tensor(
                    _flatten_singleton_batch(
                        encoded[mask_key],
                        field_name="Multimodal apply_chat_template assistant mask",
                    ),
                    dtype=torch.long,
                )
            else:
                encoding = tokenizer(
                    formatted_raw,
                    return_offsets_mapping=True,
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=False,
                )

                input_ids = _flatten_singleton_batch(
                    encoding["input_ids"], field_name="Multimodal tokenizer input_ids"
                )
                offsets = _flatten_singleton_batch(
                    encoding["offset_mapping"],
                    field_name="Multimodal tokenizer offset_mapping",
                )
                loss_mask = _create_loss_mask_from_offsets(
                    formatted_raw, offsets, assistant_pattern
                )

            assert len(input_ids) == len(loss_mask), (
                f"Shape mismatch: input_ids={len(input_ids)}, "
                f"loss_mask={len(loss_mask)}"
            )

            if minimum_valid_tokens is not None:
                num_valid_tokens = int(loss_mask.sum().item())
                if num_valid_tokens < minimum_valid_tokens:
                    continue

            mm_data = _extract_multimodal_data_from_conversation(normalized_conv)

            results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            results["loss_mask"].append(loss_mask)
            results["multi_modal_data"].append(mm_data)
            results["prompt"].append(formatted_raw)
            results["seq_len"].append(len(input_ids))

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            log.error(
                f"Failed to process conversation {idx} "
                f"(assistant_pattern={assistant_pattern is not None}): {e}"
            )
            continue

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
    num_proc: int = 8,
    assistant_pattern: str | Pattern[str] | None = None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    processor: Any | None = None,
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
        minimum_valid_tokens: Number of tokens to consider for a valid sample
    """
    is_multimodal = processor is not None
    caller = processor if is_multimodal else tokenizer

    # Detect and use provided assistant message pattern
    if assistant_pattern is not None:
        log.info(f"Using custom assistant pattern: {str(assistant_pattern)[:80]}...")
    elif _supports_assistant_mask(caller):
        assistant_pattern = None  # Signal to use HF mask in _preprocess_batch
        suffix = " (multimodal)" if is_multimodal else ""
        log.info(f"Using HF assistant token mask for loss masking{suffix}")
    else:
        assistant_pattern = _detect_assistant_pattern(caller)
        log.info(f"Detected assistant pattern: {str(assistant_pattern)[:80]}...")

    original_cols = dataset.column_names

    if is_multimodal:
        map_fn = partial(
            _preprocess_batch_multimodal,
            processor=processor,
            max_length=max_length,
            assistant_pattern=assistant_pattern,
            turn_dropout=turn_dropout,
            minimum_valid_tokens=minimum_valid_tokens,
        )
    else:
        map_fn = partial(
            _preprocess_batch,
            tokenizer=tokenizer,
            max_length=max_length,
            assistant_pattern=assistant_pattern,
            turn_dropout=turn_dropout,
            minimum_valid_tokens=minimum_valid_tokens,
        )

    dataset = dataset.map(
        map_fn,
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        keep_in_memory=True,  # skip caching
    )

    if is_multimodal:
        dataset.set_format(
            type="torch", columns=["input_ids", "loss_mask"], output_all_columns=True
        )
    else:
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


def load_and_preprocess_dataset(  # noqa: PLR0912
    target_model_path: str,
    train_data_paths: list[str],
    seq_length: int,
    build_dataset_num_proc: int = 8,
    seed: int = 0,
    max_samples: int | None = None,
    token_freq_path: Path | str = "./token_freq.pt",  # noqa: S107
    assistant_pattern: str | None = None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    is_multimodal: bool | None = None,
) -> tuple[HFDataset, PreTrainedTokenizerBase]:
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
        minimum_valid_tokens: Number of tokens to consider for a valid sample

    Returns:
        Tuple of (preprocessed_dataset, tokenizer)
    """
    if minimum_valid_tokens is not None and minimum_valid_tokens < 0:
        raise ValueError("minimum_valid_tokens must be >= 0")
    log.section("Starting dataset preprocessing")
    if minimum_valid_tokens is not None:
        log.info(
            f"Filtering samples with fewer than {minimum_valid_tokens} valid tokens"
        )

    if is_multimodal is None:
        is_multimodal = False
    log.info(f"Using multimodal mode: {is_multimodal}")

    log.subsection("Loading tokenizer")
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(
            target_model_path, trust_remote_code=True
        )
        tokenizer = _get_tokenizer_from_processor(processor)
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            target_model_path, trust_remote_code=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_multimodal:
        if not hasattr(processor, "apply_chat_template"):
            raise ValueError(
                f"Processor for {target_model_path} does not support "
                "apply_chat_template. Please use a model with a pre-configured "
                "chat template."
            )
    elif (
        not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None
    ):
        raise ValueError(
            f"Tokenizer for {target_model_path} does not support chat templates. "
            "Please use a model with a pre-configured chat template."
        )

    processed_datasets = []
    for train_data_path in train_data_paths:
        log.subsection(f"Processing {train_data_path}")
        raw_dataset = load_raw_dataset(train_data_path, num_proc=build_dataset_num_proc)
        raw_dataset = raw_dataset.shuffle(seed=seed)

        if max_samples is not None and len(raw_dataset) > 3 * max_samples:
            # Reduce size to 3 * max_samples to reduce processing
            # This will then be reduced further to max_samples
            # after combining datasets and shuffling
            raw_dataset = raw_dataset.select(range(3 * max_samples))

        log.info(f"Loaded {len(raw_dataset)} samples")

        if turn_dropout:
            log.info("Turn dropout enabled: randomly keeping N consecutive turns")

        preprocessed_dataset = build_eagle3_dataset(
            dataset=raw_dataset,
            tokenizer=tokenizer,
            max_length=seq_length,
            num_proc=build_dataset_num_proc,
            assistant_pattern=assistant_pattern,
            turn_dropout=turn_dropout,
            minimum_valid_tokens=minimum_valid_tokens,
            processor=processor,
        )
        if minimum_valid_tokens is not None:
            log.info(f"Kept {len(preprocessed_dataset)} samples after filtering")
        processed_datasets.append(preprocessed_dataset)

    combined_dataset = concatenate_datasets(processed_datasets)
    combined_dataset = combined_dataset.shuffle(seed=seed)
    if max_samples is not None and len(combined_dataset) > max_samples:
        combined_dataset = combined_dataset.select(range(max_samples))

    log.subsection("Computing token frequency distribution")
    save_token_frequency_distribution(
        dataset=combined_dataset,
        output_path=token_freq_path,
    )

    log.subsection("Visualizing sample")
    _visualize_sample(combined_dataset, tokenizer, idx=0)

    log.section("Dataset preprocessing complete")

    return combined_dataset, tokenizer
