import bisect
import random
import re
from collections.abc import Callable
from pathlib import Path
from re import Pattern
from typing import cast

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoProcessor,
    BatchEncoding,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


ProcessorLike = PreTrainedTokenizerBase | ProcessorMixin


def _visualize_sample(preprocessed: HFDataset, processor: ProcessorLike, idx: int = 0):
    """Visualize a single sample with color-coded trainable regions."""
    # Get preprocessed sample
    prep_sample = preprocessed[idx]
    input_ids = prep_sample["input_ids"].tolist()
    loss_mask = prep_sample["loss_mask"].tolist()

    log.info(f"SAMPLE #{idx}")
    log.info("HIGHLIGHTED TEXT (BLUE = trainable, GREY = masked)")

    # Create color-highlighted text
    blue = "\033[38;5;153m"  # Very light blue text for trainable tokens
    grey = "\033[90m"  # Grey text for masked tokens
    reset = "\033[0m"  # Reset color

    output = []
    prev_state = None

    for i in range(len(input_ids)):
        is_train = loss_mask[i] == 1
        token = processor.decode([input_ids[i]])
        assert isinstance(token, str)

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
        content = turn.get("value") or turn.get("content") or ""

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


def _hf_to_vllm_part(part: str | dict):
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]

    if part_type == "text":
        return {"type": "text", "text": part["text"]}

    for modality in ("image", "video", "audio"):
        if part_type == modality:
            if local_path := part.get("path"):
                file_url = f"file://{Path(local_path).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}
            if url := part.get("url"):
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}

            fields_expr = {f"part.{k}" for k in part if k != "type"}

            raise NotImplementedError(
                f"No handler defined in part.type={part_type!r} "
                f"for fields: {fields_expr}"
            )

    raise NotImplementedError(f"No handler defined for part.type={part_type!r}")


def _hf_to_vllm_turn(turn: dict):
    if isinstance(turn["content"], str):
        return turn

    return turn | {"content": [_hf_to_vllm_part(part) for part in turn["content"]]}


def _hf_to_vllm_conv(normalized_conv: list[dict]):
    return [_hf_to_vllm_turn(turn) for turn in normalized_conv]


def _get_assistant_mask_test_conv(processor: ProcessorLike):
    if isinstance(processor, ProcessorMixin):
        return [{"role": "assistant", "content": [{"type": "text", "text": "test"}]}]

    return [{"role": "assistant", "content": "test"}]


def _supports_assistant_mask(processor: ProcessorLike) -> bool:
    """Check if processor truly supports HF assistant token mask.

    Must return a non-zero mask for a conversation containing an assistant message.
    """
    test_conv = _get_assistant_mask_test_conv(processor)

    try:
        res_any = processor.apply_chat_template(
            test_conv,
            tokenizer=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        res = cast("BatchEncoding | BatchFeature", res_any)

        # Check both singular and plural key names
        mask = res.get("assistant_masks", res.get("assistant_mask"))
        if mask is None:
            return False

        # Verify the mask is not all zeros
        return any(m == 1 for m in mask)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        log.warning(f"An error occurred when trying to return assistant mask: {e}")
        return False


def _get_assistant_pattern_test_conv(processor: ProcessorLike):
    if isinstance(processor, ProcessorMixin):
        return [
            {"role": "user", "content": [{"type": "text", "text": "USER_MSG_1"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ASSISTANT_MSG_1"}],
            },
            {"role": "user", "content": [{"type": "text", "text": "USER_MSG_2"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ASSISTANT_MSG_2"}],
            },
        ]

    return [
        {"role": "user", "content": "USER_MSG_1"},
        {"role": "assistant", "content": "ASSISTANT_MSG_1"},
        {"role": "user", "content": "USER_MSG_2"},
        {"role": "assistant", "content": "ASSISTANT_MSG_2"},
    ]


def _detect_assistant_pattern(processor: ProcessorLike) -> str:
    """Auto-detect the assistant message pattern from the processor's chat template.

    Uses multi-turn conversation but extracts pattern from the LAST assistant
    message only.
    """
    test_conv = _get_assistant_pattern_test_conv(processor)

    formatted = processor.apply_chat_template(
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


def _get_input_ids_loss_mask(
    normalized_conv: list[dict],
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
):
    if assistant_pattern is None:
        # HF assistant token mask
        encoded_any = processor.apply_chat_template(
            normalized_conv,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        encoded = cast("BatchEncoding | BatchFeature", encoded_any)

        # input IDs and loss mask
        input_ids = encoded["input_ids"]
        # HF uses 'assistant_masks' in recent versions
        mask_key = (
            "assistant_masks" if "assistant_masks" in encoded else "assistant_mask"
        )
        loss_mask = torch.tensor(encoded[mask_key], dtype=torch.long)

        return input_ids, loss_mask

    # Fallback: regex-based detection
    assert assistant_pattern is not None, "Assistant pattern required for fallback"

    processor_kwargs: dict = {
        "return_offsets_mapping": True,
        "max_length": max_length,
        "truncation": True,
        "add_special_tokens": False,
    }

    if isinstance(processor, ProcessorMixin):
        encoded_any = processor.apply_chat_template(
            normalized_conv,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            processor_kwargs=processor_kwargs,
        )
        encoded = cast("BatchFeature", encoded_any)

        # Remove batch dimension
        (input_ids,) = encoded["input_ids"]
        (offsets,) = encoded["offset_mapping"]

        # MM placeholder tokens are inserted separate from chat template
        formatted_text = processor.decode(input_ids)
        assert isinstance(formatted_text, str)
    else:
        # More optimized flow for text-only processors (i.e. tokenizers)
        formatted_text = processor.apply_chat_template(
            normalized_conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(formatted_text, str)

        # Tokenize and get offsets
        encoded_any = processor(formatted_text, **processor_kwargs)
        encoded = cast("BatchEncoding", encoded_any)

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

    loss_mask = _create_loss_mask_from_offsets(
        formatted_text, offsets, assistant_pattern
    )

    return input_ids, loss_mask


def _preprocess_batch(
    examples: dict,
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    conversations: list[dict] = examples.get("conversations", [])

    # MM inputs must use Chat Completion API
    if isinstance(processor, ProcessorMixin):
        results["_vllm_messages"] = []

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
            input_ids, loss_mask = _get_input_ids_loss_mask(
                normalized_conv,
                processor,
                max_length=max_length,
                assistant_pattern=assistant_pattern,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            log.error(
                f"Failed to process conversation {idx} "
                f"(assistant_pattern={assistant_pattern is not None}): {e}"
            )
            continue

        # Assert shapes match
        assert len(input_ids) == len(loss_mask), (
            f"Shape mismatch: input_ids={len(input_ids)}, loss_mask={len(loss_mask)}"
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

        if "_vllm_messages" in results:
            results["_vllm_messages"].append(_hf_to_vllm_conv(normalized_conv))

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    processor: ProcessorLike,
    max_length: int = 2048,
    num_proc: int = 8,
    assistant_pattern: str | Pattern[str] | None = None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
) -> HFDataset:
    """Build EAGLE3 dataset by tokenizing conversations and creating loss masks.

    Uses the processor's built-in chat template via apply_chat_template.

    Args:
        dataset: Raw dataset with conversations
        processor: Processor with chat template support
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel processing
        assistant_pattern: Optional custom regex pattern for matching assistant
                          responses. If None, pattern will be auto-detected from
                          chat template.
        turn_dropout: If True, randomly keeps first N consecutive turns per
                     conversation
        minimum_valid_tokens: Number of tokens to consider for a valid sample
    """
    # Detect and use provided assistant message pattern
    if assistant_pattern is not None:
        log.info(f"Using custom assistant pattern: {str(assistant_pattern)[:80]}...")
    elif _supports_assistant_mask(processor):
        assistant_pattern = None  # Signal to use HF mask in _preprocess_batch
        log.info("Using HF assistant token mask for loss masking")
    else:
        assistant_pattern = _detect_assistant_pattern(processor)
        log.info(f"Detected assistant pattern: {str(assistant_pattern)[:80]}...")

    original_cols = dataset.column_names

    dataset = dataset.map(
        lambda examples: _preprocess_batch(
            examples,
            processor,
            max_length,
            assistant_pattern,
            turn_dropout,
            minimum_valid_tokens,
        ),
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        keep_in_memory=True,  # skip caching
    )

    dataset.set_format(type="torch")
    return dataset


def load_raw_dataset(
    train_data_path: str,
) -> tuple[HFDataset, Callable[[dict], dict] | None]:
    """Load raw dataset from local file or HuggingFace."""
    if train_data_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=train_data_path, split="train"), None

    if train_data_path not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {train_data_path}. "
            f"Supported: local .json/.jsonl files or {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[train_data_path]
    raw_dataset = load_dataset(config.hf_path, config.hf_name, split=config.split)

    if config.filter_fn is not None:
        raw_dataset = raw_dataset.filter(config.filter_fn)

    return raw_dataset, config.normalize_fn


def _resolve_eos_token(processor: ProcessorLike):
    tokenizer = (
        processor.tokenizer if isinstance(processor, ProcessorMixin) else processor
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_processor(target_model_path: str, *, trust_remote_code: bool = False):
    processor = AutoProcessor.from_pretrained(
        target_model_path,
        trust_remote_code=trust_remote_code,
    )
    _resolve_eos_token(processor)

    return processor


def load_and_preprocess_dataset(
    target_model_path: str,
    train_data_paths: list[str],
    *,
    seq_length: int,
    build_dataset_num_proc: int = 8,
    seed: int = 0,
    max_samples: int | None = None,
    token_freq_path: Path | str = "./token_freq.pt",  # noqa: S107
    assistant_pattern: str | None = None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    trust_remote_code: bool = False,
) -> tuple[HFDataset, ProcessorLike]:
    """Load, tokenize, and preprocess a dataset for EAGLE3 training.

    Uses the processor's built-in chat template via apply_chat_template.
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
        trust_remote_code: If True, allows executing code from HF Hub.

    Returns:
        Tuple of (preprocessed_dataset, processor)
    """
    if minimum_valid_tokens is not None and minimum_valid_tokens < 0:
        raise ValueError("minimum_valid_tokens must be >= 0")
    log.section("Starting dataset preprocessing")
    if minimum_valid_tokens is not None:
        log.info(
            f"Filtering samples with fewer than {minimum_valid_tokens} valid tokens"
        )

    log.subsection("Loading processor")
    processor = _load_processor(target_model_path, trust_remote_code=trust_remote_code)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        raise ValueError(
            f"Processor for {target_model_path} does not support chat templates. "
            "Please use a model with a pre-configured chat template."
        )

    processed_datasets = []
    for train_data_path in train_data_paths:
        log.subsection(f"Processing {train_data_path}")
        raw_dataset, normalize_fn = load_raw_dataset(train_data_path)
        raw_dataset = raw_dataset.shuffle(seed=seed)

        if max_samples is not None and len(raw_dataset) > 3 * max_samples:
            # Reduce size to 3 * max_samples to reduce processing
            # This will then be reduced further to max_samples
            # after combining datasets and shuffling
            raw_dataset = raw_dataset.select(range(3 * max_samples))

        if normalize_fn is not None:
            raw_dataset = raw_dataset.map(
                normalize_fn,
                num_proc=build_dataset_num_proc,
                keep_in_memory=True,  # skip caching
            )

        log.info(f"Loaded {len(raw_dataset)} samples")

        if turn_dropout:
            log.info("Turn dropout enabled: randomly keeping N consecutive turns")

        preprocessed_dataset = build_eagle3_dataset(
            dataset=raw_dataset,
            processor=processor,
            max_length=seq_length,
            num_proc=build_dataset_num_proc,
            assistant_pattern=assistant_pattern,
            turn_dropout=turn_dropout,
            minimum_valid_tokens=minimum_valid_tokens,
        )
        if minimum_valid_tokens is not None:
            log.info(f"Kept {len(preprocessed_dataset)} samples after filtering")
        processed_datasets.append(preprocessed_dataset)

    combined_dataset = concatenate_datasets(processed_datasets)
    combined_dataset.shuffle(seed=seed)
    if max_samples is not None and len(combined_dataset) > max_samples:
        combined_dataset = combined_dataset.select(range(max_samples))

    log.subsection("Computing token frequency distribution")
    save_token_frequency_distribution(
        dataset=combined_dataset,
        output_path=token_freq_path,
    )

    log.subsection("Visualizing sample")
    _visualize_sample(combined_dataset, processor, idx=0)

    log.section("Dataset preprocessing complete")

    return combined_dataset, processor
