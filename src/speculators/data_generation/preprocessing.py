import bisect
import inspect
import json
import random
import re
from pathlib import Path
from re import Pattern
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
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

MULTIMODAL_SIDECAR_DIR = "multimodal"
MULTIMODAL_MEDIA_TYPES = {"image", "video", "audio"}
MULTIMODAL_ENCODER_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grids",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
)


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


def _normalize_media_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    filename = getattr(value, "filename", None)
    if filename:
        return str(filename)
    return value


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    filename = getattr(value, "filename", None)
    if filename:
        return str(filename)
    return value


def _normalize_content_segment(segment: Any) -> dict[str, Any]:
    if isinstance(segment, str):
        return {"type": "text", "text": segment}
    if not isinstance(segment, dict):
        return {"type": "text", "text": str(segment)}

    seg_type = str(segment.get("type", "text"))
    normalized = {"type": seg_type}

    if seg_type == "text":
        normalized["text"] = (
            segment.get("text")
            or segment.get("value")
            or segment.get("content")
            or ""
        )
    elif seg_type in MULTIMODAL_MEDIA_TYPES:
        normalized[seg_type] = _normalize_media_value(
            segment.get(seg_type)
            or segment.get("value")
            or segment.get("url")
            or segment.get("path")
            or segment.get("source")
        )
    else:
        normalized["text"] = segment.get("text") or segment.get("value") or ""

    for key, value in segment.items():
        if key in normalized or key in {"content", "value", "url", "path", "source"}:
            continue
        normalized[key] = _to_json_compatible(value)

    return normalized


def _normalize_turn_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, list):
        return [_normalize_content_segment(seg) for seg in content]
    return content if isinstance(content, str) else str(content or "")


def _has_multimodal_segments(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(seg, dict) and seg.get("type") in MULTIMODAL_MEDIA_TYPES
        for seg in content
    )


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(_to_json_compatible(messages), ensure_ascii=False)


def _sanitize_sidecar_prefix(prefix: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", prefix).strip("_") or "dataset"


def _build_sidecar_path(
    multimodal_output_dir: str | Path,
    sample_idx: int,
    sidecar_prefix: str,
) -> Path:
    base_dir = Path(multimodal_output_dir) / MULTIMODAL_SIDECAR_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = _sanitize_sidecar_prefix(sidecar_prefix)
    return base_dir / f"{safe_prefix}_{sample_idx}.safetensors"


def _maybe_strip_batch_dim(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _save_multimodal_sidecar(
    encoded: dict[str, Any],
    multimodal_output_dir: str | Path,
    sample_idx: int,
    sidecar_prefix: str,
) -> str:
    sidecar_path = _build_sidecar_path(multimodal_output_dir, sample_idx, sidecar_prefix)
    payload = {
        key: _maybe_strip_batch_dim(encoded[key])
        for key in MULTIMODAL_ENCODER_KEYS
        if key in encoded
    }
    if not payload:
        raise ValueError("Multimodal sample did not produce any sidecar tensor fields")
    save_file(payload, sidecar_path)
    return str(sidecar_path.relative_to(Path(multimodal_output_dir)))


def _build_multimodal_loss_mask(
    input_ids: torch.Tensor,
    base_loss_mask: torch.Tensor,
    placeholder_token_ids: tuple[int, ...],
) -> torch.Tensor:
    loss_mask = base_loss_mask.to(dtype=torch.long).clone()
    for token_id in placeholder_token_ids:
        if token_id >= 0:
            loss_mask[input_ids == token_id] = 0
    return loss_mask


_PROCESSOR_KW_CACHE: dict[int, set[str]] = {}


def _processor_kwargs(processor: Any) -> set[str]:
    """Discover which kwargs ``processor.apply_chat_template`` actually accepts.

    Older transformers versions do not support ``load_image / load_audio /
    load_video`` nor ``return_assistant_tokens_mask`` on the processor-level
    ``apply_chat_template``; passing them unconditionally either (a) silently
    no-ops or (b) raises ``TypeError`` depending on the version. We therefore
    probe the signature once per processor instance and only forward kwargs
    the callee declares.
    """
    key = id(processor)
    cached = _PROCESSOR_KW_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(processor.apply_chat_template)
        names = set(sig.parameters.keys())
        has_varkw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_varkw:
            # **kwargs wildcard: treat all "known" multimodal kwargs as allowed.
            names |= {
                "load_audio",
                "load_image",
                "load_video",
                "load_audios",
                "load_images",
                "load_videos",
                "return_assistant_tokens_mask",
            }
    except (TypeError, ValueError):
        names = set()
    _PROCESSOR_KW_CACHE[key] = names
    return names


def _conversation_use_audio_in_video(conv: list[dict[str, Any]]) -> bool:
    for turn in conv:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for seg in content:
            if isinstance(seg, dict) and seg.get("use_audio_in_video"):
                return True
    return False


def _is_multimodal_batch(examples: dict) -> bool:
    """Detect whether a batch contains content-list style multimodal samples."""
    convs = examples.get("conversations", [])
    if not convs:
        return False
    for conv in convs:
        for turn in conv or []:
            if _has_multimodal_segments(turn.get("content") or turn.get("value")):
                return True
    return False


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
        content = _normalize_turn_content(turn.get("value") or turn.get("content") or "")

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


def _supports_assistant_mask(tokenizer: PreTrainedTokenizerBase) -> bool:
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


def _detect_assistant_pattern(tokenizer: PreTrainedTokenizerBase) -> str:
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


def _loss_mask_from_ids_fallback(
    input_ids: torch.Tensor,
    normalized_conv: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    assistant_pattern: str | Pattern[str],
) -> torch.Tensor:
    """Best-effort assistant-token mask for multimodal samples when the
    processor does not natively emit ``assistant_masks``.

    Strategy: re-render the conversation through the **tokenizer** chat template
    (text-only), grab character offsets via
    ``tokenizer(..., return_offsets_mapping=True)``, locate assistant spans via
    ``assistant_pattern``, then **left-align** the resulting mask to
    ``input_ids`` using the lengths from the processor path. Any length
    mismatch is resolved by truncating / zero-padding to ``input_ids.shape[0]``.

    This is inherently approximate (processor vs tokenizer may tokenize media
    placeholders slightly differently), but is strictly better than the
    previous behaviour of raising and dropping the whole sample.
    """
    formatted_raw_any = tokenizer.apply_chat_template(
        normalized_conv,
        tokenize=False,
        add_generation_prompt=False,
    )
    formatted_raw = formatted_raw_any if isinstance(formatted_raw_any, str) else ""
    encoding = tokenizer(
        formatted_raw,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]
    mask_text = _create_loss_mask_from_offsets(
        formatted_raw, offsets, assistant_pattern
    ).to(torch.long)

    target_len = int(input_ids.shape[0])
    if mask_text.shape[0] == target_len:
        return mask_text
    if mask_text.shape[0] > target_len:
        return mask_text[:target_len]
    pad = torch.zeros(target_len - mask_text.shape[0], dtype=torch.long)
    return torch.cat([mask_text, pad], dim=0)


def _preprocess_batch(
    examples: dict,
    indices: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    processor: Any | None = None,
    placeholder_token_ids: tuple[int, ...] = (),
    multimodal_output_dir: str | Path | None = None,
    sidecar_prefix: str = "dataset",
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    results: dict[str, list] = {
        "input_ids": [],
        "loss_mask": [],
        "seq_len": [],
        "mm_file": [],
        "messages_json": [],
        "use_audio_in_video": [],
    }
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

        is_multimodal = _is_multimodal_batch({"conversations": [normalized_conv]})
        sample_idx = indices[idx]
        messages_json = _serialize_messages(normalized_conv) if is_multimodal else ""
        mm_file = ""
        use_audio_in_video = int(_conversation_use_audio_in_video(normalized_conv))

        try:
            if processor is not None and is_multimodal:
                if multimodal_output_dir is None:
                    raise ValueError(
                        "multimodal_output_dir must be set when preprocessing "
                        "multimodal samples"
                    )
                # Dynamically probe which kwargs the installed transformers
                # version's processor.apply_chat_template actually accepts.
                allowed = _processor_kwargs(processor)
                call_kwargs: dict[str, Any] = dict(
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                )
                for k in (
                    "load_audio",
                    "load_image",
                    "load_video",
                    "load_audios",
                    "load_images",
                    "load_videos",
                ):
                    if k in allowed:
                        call_kwargs[k] = True
                supports_mask = "return_assistant_tokens_mask" in allowed
                if supports_mask:
                    call_kwargs["return_assistant_tokens_mask"] = True

                encoded_any = processor.apply_chat_template(
                    normalized_conv, **call_kwargs
                )
                encoded = cast("dict[str, Any]", encoded_any)
                input_ids = _maybe_strip_batch_dim(encoded["input_ids"]).to(torch.long)

                mask_key = None
                if supports_mask:
                    if "assistant_masks" in encoded:
                        mask_key = "assistant_masks"
                    elif "assistant_mask" in encoded:
                        mask_key = "assistant_mask"

                if mask_key is not None:
                    base_loss_mask = _maybe_strip_batch_dim(encoded[mask_key]).to(
                        torch.long
                    )
                else:
                    # Fallback: processor did not emit an assistant mask on
                    # this transformers version. Approximate via the tokenizer
                    # + assistant_pattern path.
                    if assistant_pattern is None:
                        # We need a pattern for the fallback; derive one now.
                        try:
                            assistant_pattern = _detect_assistant_pattern(tokenizer)
                        except (ValueError, KeyError, AttributeError):
                            raise ValueError(  # noqa: B904
                                "Processor did not return assistant_masks and "
                                "no assistant_pattern fallback could be "
                                "auto-detected for this tokenizer."
                            )
                    base_loss_mask = _loss_mask_from_ids_fallback(
                        input_ids,
                        normalized_conv,
                        tokenizer,
                        assistant_pattern,
                    )

                loss_mask = _build_multimodal_loss_mask(
                    input_ids,
                    base_loss_mask,
                    placeholder_token_ids,
                )
                mm_file = _save_multimodal_sidecar(
                    encoded,
                    multimodal_output_dir,
                    sample_idx,
                    sidecar_prefix,
                )
            elif assistant_pattern is None:
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
                input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
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
                input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
                offsets = encoding["offset_mapping"]

                loss_mask = _create_loss_mask_from_offsets(
                    formatted_raw, offsets, assistant_pattern
                ).to(torch.long)

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
            results["input_ids"].append(input_ids)
            results["loss_mask"].append(loss_mask)
            results["seq_len"].append(len(input_ids))
            results["mm_file"].append(mm_file)
            results["messages_json"].append(messages_json)
            results["use_audio_in_video"].append(use_audio_in_video)

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
    placeholder_token_ids: tuple[int, ...] = (),
    multimodal_output_dir: str | Path | None = None,
    sidecar_prefix: str = "dataset",
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
    # Detect and use provided assistant message pattern
    if assistant_pattern is not None:
        log.info(f"Using custom assistant pattern: {str(assistant_pattern)[:80]}...")
    elif _supports_assistant_mask(tokenizer):
        assistant_pattern = None  # Signal to use HF mask in _preprocess_batch
        log.info("Using HF assistant token mask for loss masking")
    else:
        assistant_pattern = _detect_assistant_pattern(tokenizer)
        log.info(f"Detected assistant pattern: {str(assistant_pattern)[:80]}...")

    if multimodal_output_dir is not None:
        (Path(multimodal_output_dir) / MULTIMODAL_SIDECAR_DIR).mkdir(
            parents=True, exist_ok=True
        )

    original_cols = dataset.column_names

    dataset = dataset.map(
        lambda examples, indices: _preprocess_batch(
            examples,
            indices,
            tokenizer,
            max_length,
            assistant_pattern,
            turn_dropout,
            minimum_valid_tokens,
            processor=processor,
            placeholder_token_ids=placeholder_token_ids,
            multimodal_output_dir=multimodal_output_dir,
            sidecar_prefix=sidecar_prefix,
        ),
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        keep_in_memory=True,  # skip caching
    )

    dataset.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
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


def load_and_preprocess_dataset(
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
    multimodal: bool = False,
    multimodal_output_dir: Path | str | None = None,
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
    if multimodal and multimodal_output_dir is None:
        raise ValueError(
            "multimodal_output_dir must be provided when multimodal preprocessing "
            "is enabled"
        )

    log.section("Starting dataset preprocessing")
    if minimum_valid_tokens is not None:
        log.info(
            f"Filtering samples with fewer than {minimum_valid_tokens} valid tokens"
        )

    log.subsection("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {target_model_path} does not support chat templates. "
            "Please use a model with a pre-configured chat template."
        )

    processor = None
    placeholder_token_ids: tuple[int, ...] = ()
    verifier_config = AutoConfig.from_pretrained(
        target_model_path, trust_remote_code=True
    )
    multimodal_config = getattr(verifier_config, "thinker_config", verifier_config)
    if multimodal:
        processor = AutoProcessor.from_pretrained(
            target_model_path, trust_remote_code=True
        )
        placeholder_token_ids = tuple(
            int(token_id)
            for attr in ("image_token_id", "video_token_id", "audio_token_id")
            if (token_id := getattr(multimodal_config, attr, None)) is not None
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
            placeholder_token_ids=placeholder_token_ids,
            multimodal_output_dir=multimodal_output_dir,
            sidecar_prefix=train_data_path,
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
