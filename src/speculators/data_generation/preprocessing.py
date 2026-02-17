import bisect
import random
import re
from re import Pattern
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "create_loss_mask_from_token_ids",
    "detect_assistant_token_markers",
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


def _supports_assistant_mask(tokenizer: PreTrainedTokenizer) -> bool:
    """Check if tokenizer truly supports HF assistant token mask.

    Must return a non-zero mask for a conversation containing an assistant message.
    """
    try:
        res_any = tokenizer.apply_chat_template(
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
        return any(m == 1 for m in mask)
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


def _find_subsequence(
    sequence: list[int], subsequence: list[int], start: int = 0
) -> int:
    """Find the first occurrence of subsequence in sequence starting from start.

    Returns index or -1.
    """
    sub_len = len(subsequence)
    for i in range(start, len(sequence) - sub_len + 1):
        if sequence[i : i + sub_len] == subsequence:
            return i
    return -1


def detect_assistant_token_markers(
    tokenizer: PreTrainedTokenizer,
) -> tuple[list[int], list[int]]:
    """Detect the token ID sequences that mark assistant turn boundaries.

    Uses test conversations to locate content tokens and extract the structural
    tokens that bracket assistant content. Works across all major chat template
    families (ChatML, Llama 3, Mistral, Phi-3, GPT-OSS, etc.).

    Args:
        tokenizer: A tokenizer with a chat template configured.

    Returns:
        (start_marker_ids, end_marker_ids): Token ID sequences that mark
        the start and end of assistant content.
    """
    # Use multi-word content strings that tokenize independently
    # (avoids tokenizer merging single chars with adjacent special tokens)
    conv_long = [
        {"role": "user", "content": "alpha bravo"},
        {"role": "assistant", "content": "charlie delta"},
        {"role": "user", "content": "echo foxtrot"},
    ]
    ids_long = list(
        tokenizer.apply_chat_template(
            conv_long, tokenize=True, add_generation_prompt=False
        )
    )

    user1_ids = tokenizer.encode("alpha bravo", add_special_tokens=False)
    asst_ids = tokenizer.encode("charlie delta", add_special_tokens=False)
    user2_ids = tokenizer.encode("echo foxtrot", add_special_tokens=False)

    # Find positions of each content block (in order)
    user1_pos = _find_subsequence(ids_long, user1_ids)
    if user1_pos == -1:
        raise ValueError("Could not find user1 content tokens in templated sequence.")

    asst_pos = _find_subsequence(ids_long, asst_ids, user1_pos + len(user1_ids))
    if asst_pos == -1:
        raise ValueError(
            "Could not find assistant content tokens in templated sequence."
        )

    user2_pos = _find_subsequence(ids_long, user2_ids, asst_pos + len(asst_ids))
    if user2_pos == -1:
        raise ValueError("Could not find user2 content tokens in templated sequence.")

    # Start marker = tokens between user1 content end and assistant content start
    start_marker_ids = ids_long[user1_pos + len(user1_ids) : asst_pos]
    if not start_marker_ids:
        raise ValueError("No tokens found between user and assistant content.")

    # End marker extraction:
    # Compare the 3-turn (long) conversation with a 2-turn (short) one.
    # Both share the same tokens up to and including the assistant content
    # plus the end-of-turn marker. They diverge after that.
    asst_end = asst_pos + len(asst_ids)

    conv_short = [
        {"role": "user", "content": "alpha bravo"},
        {"role": "assistant", "content": "charlie delta"},
    ]
    ids_short = list(
        tokenizer.apply_chat_template(
            conv_short, tokenize=True, add_generation_prompt=False
        )
    )

    # Find longest common prefix between short and long sequences
    common_len = 0
    for i in range(min(len(ids_short), len(ids_long))):
        if ids_short[i] == ids_long[i]:
            common_len = i + 1
        else:
            break

    if common_len > asst_end:
        # Shared tokens after assistant content = end marker
        end_marker_ids = ids_long[asst_end:common_len]
    else:
        # Short and long conversations use different end-of-turn tokens
        # (e.g., GPT-OSS uses <|return|> for final turn, <|end|> for mid-turn).
        # Use the first token after assistant content in the long conversation
        # as the end marker — this is the mid-turn end token.
        end_marker_ids = ids_long[asst_end : asst_end + 1]

    if not end_marker_ids:
        raise ValueError("Could not determine assistant end marker tokens.")

    # Validate: start marker should appear in multi-turn conversations
    multi_turn = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    multi_ids = list(
        tokenizer.apply_chat_template(
            multi_turn, tokenize=True, add_generation_prompt=False
        )
    )

    count = 0
    pos = 0
    while True:
        found = _find_subsequence(multi_ids, start_marker_ids, pos)
        if found == -1:
            break
        count += 1
        pos = found + 1

    if count < 2:
        log.warning(
            f"Expected start marker to appear >= 2 times in multi-turn "
            f"conversation, found {count}. Masking may be inaccurate."
        )

    log.info(
        f"Detected assistant start marker: {start_marker_ids} "
        f"({len(start_marker_ids)} tokens)"
    )
    log.info(
        f"Detected assistant end marker: {end_marker_ids} "
        f"({len(end_marker_ids)} tokens)"
    )

    return start_marker_ids, end_marker_ids


def create_loss_mask_from_token_ids(
    token_ids: list[int] | torch.Tensor,
    start_marker_ids: list[int],
    end_marker_ids: list[int],
) -> torch.Tensor:
    """Create binary loss mask by scanning token IDs for assistant turn markers.

    Tokens within assistant turns (between start and end markers) get mask=1.
    Marker tokens themselves get mask=0. All other tokens get mask=0.

    Handles multiple assistant turns, truncated sequences (no end marker marks
    to end of sequence), and system messages (naturally excluded).

    Args:
        token_ids: Full token ID sequence.
        start_marker_ids: Token sequence marking start of assistant content.
        end_marker_ids: Token sequence marking end of assistant content.

    Returns:
        Binary tensor of same length as token_ids (dtype=torch.long).
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    n = len(token_ids)
    mask = [0] * n
    start_len = len(start_marker_ids)
    end_len = len(end_marker_ids)

    i = 0
    while i < n:
        # Check for start marker
        if token_ids[i : i + start_len] == start_marker_ids:
            # Skip past the start marker (don't include it in mask)
            j = i + start_len
            # Mark tokens until end marker or end of sequence
            while j < n:
                if token_ids[j : j + end_len] == end_marker_ids:
                    j += end_len  # skip past end marker
                    break
                mask[j] = 1
                j += 1
            i = j
        else:
            i += 1

    return torch.tensor(mask, dtype=torch.long)


def _preprocess_batch(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
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
                input_ids = encoded["input_ids"]
                # HF uses 'assistant_masks' in recent versions
                mask_key = (
                    "assistant_masks"
                    if "assistant_masks" in encoded
                    else "assistant_mask"
                )
                loss_mask = torch.tensor(encoded[mask_key], dtype=torch.long)

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
            log.error(
                f"Failed to process conversation {idx} "
                f"(assistant_pattern={assistant_pattern is not None}): {e}"
            )
            continue

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    num_proc: int = 8,
    assistant_pattern: str | Pattern[str] | None = None,
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
    if assistant_pattern is not None:
        log.info(f"Using custom assistant pattern: {str(assistant_pattern)[:80]}...")
    elif _supports_assistant_mask(tokenizer):
        assistant_pattern = None  # Signal to use HF mask in _preprocess_batch
        log.info("Using HF assistant token mask for loss masking")
    else:
        assistant_pattern = _detect_assistant_pattern(tokenizer)
        log.info(f"Detected assistant pattern: {str(assistant_pattern)[:80]}...")

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
