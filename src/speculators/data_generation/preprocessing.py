import bisect
import inspect
import json
import random
import re
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from re import Pattern
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from packaging.version import Version
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    BatchEncoding,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers import __version__ as TRANSFORMERS_VERSION  # noqa: N812

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.torch_utils import set_default_torch_num_threads
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


ProcessorLike = PreTrainedTokenizerBase | ProcessorMixin
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


def _is_multimodal_conversation(conv: list[dict[str, Any]]) -> bool:
    return any(_has_multimodal_segments(turn.get("content")) for turn in conv)


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
        raise ValueError("Multimodal sample did not produce sidecar tensor fields")
    save_file(payload, sidecar_path)
    return str(sidecar_path.relative_to(Path(multimodal_output_dir)))


def _build_multimodal_loss_mask(
    input_ids: torch.Tensor,
    base_loss_mask: torch.Tensor,
    placeholder_token_ids: tuple[int, ...],
) -> torch.Tensor:
    loss_mask = base_loss_mask.to(dtype=torch.long).clone()
    valid_ids = [tid for tid in placeholder_token_ids if tid >= 0]
    if not valid_ids:
        return loss_mask
    placeholder_tensor = torch.as_tensor(
        valid_ids, dtype=input_ids.dtype, device=input_ids.device
    )
    loss_mask.masked_fill_(torch.isin(input_ids, placeholder_tensor), 0)
    return loss_mask


def _mask_has_positive(mask: Any) -> bool:
    mask_tensor = _maybe_strip_batch_dim(mask)
    return bool(mask_tensor.numel() > 0 and torch.count_nonzero(mask_tensor).item() > 0)


_PROCESSOR_KW_CACHE: dict[int, set[str]] = {}


def _processor_kwargs(processor: Any) -> set[str]:
    key = id(processor)
    cached = _PROCESSOR_KW_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(processor.apply_chat_template)
        names = {
            name
            for name, param in sig.parameters.items()
            if param.kind is not inspect.Parameter.VAR_KEYWORD
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


def _as_processor_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [
            seg if isinstance(seg, dict) else {"type": "text", "text": str(seg)}
            for seg in content
        ]
    return [
        {"type": "text", "text": content if isinstance(content, str) else str(content or "")}
    ]


def _conversation_for_processor(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**turn, "content": _as_processor_content_blocks(turn.get("content", ""))}
        for turn in conv
    ]


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
        elif role == "tool":
            role = "tool"
        else:
            log.warning(f"Unknown role '{role}', skipping turn")
            continue

        # Build normalized turn with role and content
        normalized_turn = {"role": role, "content": content}

        # Preserve tool_calls and tool_call_id if present
        if turn.get("tool_calls"):
            normalized_turn["tool_calls"] = turn["tool_calls"]
        if turn.get("tool_call_id"):
            normalized_turn["tool_call_id"] = turn["tool_call_id"]

        thinking = turn.get("thinking") or turn.get("reasoning_content")
        if thinking:
            normalized_turn["thinking"] = thinking
            normalized_turn["reasoning_content"] = thinking

        normalized.append(normalized_turn)

        # Stop if we've reached the truncation point
        if i + 1 >= num_turns_to_keep and role == "assistant":
            # Only break after an assistant turn
            break

    return normalized


def _adapt_part_for_hf(part: str | dict, processor: ProcessorLike):
    if isinstance(part, str) and isinstance(processor, ProcessorMixin):
        return {"type": "text", "text": part}

    return part


def _adapt_turn_for_hf(turn: dict, processor: ProcessorLike):
    if isinstance(turn["content"], str):
        if isinstance(processor, ProcessorMixin):
            return turn | {"content": [_adapt_part_for_hf(turn["content"], processor)]}

        return turn

    return turn | {
        "content": [_adapt_part_for_hf(part, processor) for part in turn["content"]]
    }


def _adapt_conv_for_hf(normalized_conv: list[dict], processor: ProcessorLike):
    return [_adapt_turn_for_hf(turn, processor) for turn in normalized_conv]


def _adapt_part_for_vllm(part: str | dict):
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]

    if part_type == "text":
        return {"type": "text", "text": part["text"]}

    for modality in ("image", "video", "audio"):
        if part_type == modality:
            media_value = part.get(modality) or part.get("path") or part.get("url")
            if isinstance(media_value, str):
                if media_value.startswith(("http://", "https://", "file://")):
                    url = media_value
                else:
                    url = f"file://{Path(media_value).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}

            if part.get("base64"):
                expr = {"type": modality, "base64": "..."}
                raise ValueError(
                    f"Content part {expr} is not supported. To avoid copying "
                    f"the {modality} when saving the preprocessed dataset, "
                    f"please express {modality} inputs using file paths or URLs."
                )

            expr = {"type": modality} | {k: "..." for k in part if k != "type"}
            raise NotImplementedError(f"Unknown content part: {expr}")

    expr = dict.fromkeys(part.keys(), "...")
    raise NotImplementedError(f"Unknown content part: {expr}")


def _adapt_turn_for_vllm(turn: dict):
    if isinstance(turn["content"], str):
        return turn

    return turn | {"content": [_adapt_part_for_vllm(part) for part in turn["content"]]}


def _adapt_conv_for_vllm(normalized_conv: list[dict]):
    return [_adapt_turn_for_vllm(turn) for turn in normalized_conv]


def _supports_assistant_mask(processor: ProcessorLike) -> bool:
    """Check if processor truly supports HF assistant token mask.

    Must return a non-zero mask for a conversation containing an assistant message.
    """
    # NOTE: Some models (e.g. Qwen3.5) require a user message in the conversation,
    # even though this check only looks at the assistant turn
    test_conv = _adapt_conv_for_hf(
        [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "test"},
        ],
        processor,
    )

    try:
        res_any = processor.apply_chat_template(
            test_conv,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        res = cast("BatchEncoding | BatchFeature", res_any)

        # Check both singular and plural key names
        mask = res.get("assistant_masks", res.get("assistant_mask"))
        if mask is None:
            return False

        # Verify the mask is not all zeros
        return _mask_has_positive(mask)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        log.warning(f"An error occurred when trying to return assistant mask: {e}")
        return False


def _detect_assistant_pattern(processor: ProcessorLike) -> str:
    """Auto-detect the assistant message pattern from the processor's chat template.

    Uses multi-turn conversation but extracts pattern from the LAST assistant
    message only.
    """
    test_conv = _adapt_conv_for_hf(
        [
            {"role": "user", "content": "USER_MSG_1"},
            {"role": "assistant", "content": "ASSISTANT_MSG_1"},
            {"role": "user", "content": "USER_MSG_2"},
            {"role": "assistant", "content": "ASSISTANT_MSG_2"},
        ],
        processor,
    )

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
    *,
    # For logging
    conv_idx: int | None = None,
    max_length: int | None = None,
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
        warning_msg = "No assistant response spans found in conversation"
        if conv_idx is not None:
            warning_msg += f" {conv_idx}"

        suggestion_msg = ""
        if max_length is not None and len(offsets) == max_length:
            suggestion_msg += (
                "Consider increasing --seq-length to avoid truncating "
                "the assistant response."
            )

        log.warning(f"{warning_msg}. {suggestion_msg}")

    return loss_mask


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(seg.get("text", "")) if isinstance(seg, dict) else str(seg)
            for seg in content
        )
    return str(content or "")


def _find_token_subsequence(
    haystack: torch.Tensor,
    needle: torch.Tensor,
    start: int,
) -> int | None:
    if needle.numel() == 0 or haystack.numel() < needle.numel():
        return None
    for idx in range(start, int(haystack.numel() - needle.numel()) + 1):
        if torch.equal(haystack[idx : idx + needle.numel()], needle):
            return idx
    return None


def _loss_mask_from_assistant_token_spans(
    input_ids: torch.Tensor,
    normalized_conv: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor | None:
    loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
    cursor = 0
    matches_found = 0
    for turn in normalized_conv:
        if turn.get("role") != "assistant":
            continue
        text = _content_text(turn.get("content", ""))
        if not text:
            continue
        tokenized = tokenizer(text, add_special_tokens=False)
        token_ids = tokenized.get("input_ids", [])
        if not token_ids:
            continue
        needle = torch.as_tensor(token_ids, dtype=torch.long, device=input_ids.device)
        span_start = _find_token_subsequence(input_ids, needle, cursor)
        if span_start is None:
            log.warning("Could not align assistant content tokens in processor input_ids")
            continue
        span_end = span_start + int(needle.numel())
        loss_mask[span_start:span_end] = 1
        cursor = span_end
        matches_found += 1
    return loss_mask if matches_found else None


def _loss_mask_from_ids_fallback(
    input_ids: torch.Tensor,
    normalized_conv: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    assistant_pattern: str | Pattern[str],
    placeholder_token_ids: tuple[int, ...] = (),
) -> torch.Tensor:
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
    mask_text = _create_loss_mask_from_offsets(
        formatted_raw, encoding["offset_mapping"], assistant_pattern
    ).to(torch.long)

    target_len = int(input_ids.shape[0])
    if placeholder_token_ids:
        placeholder_tensor = torch.as_tensor(
            placeholder_token_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        is_placeholder = torch.isin(input_ids, placeholder_tensor)
        if bool(is_placeholder.any()):
            aligned = torch.zeros(target_len, dtype=torch.long)
            text_positions = (~is_placeholder).nonzero(as_tuple=True)[0]
            copy_len = min(int(text_positions.shape[0]), int(mask_text.shape[0]))
            if copy_len > 0:
                aligned.index_copy_(0, text_positions[:copy_len], mask_text[:copy_len])
            return aligned

    if mask_text.shape[0] == target_len:
        return mask_text
    if mask_text.shape[0] > target_len:
        return mask_text[:target_len]
    pad = torch.zeros(target_len - mask_text.shape[0], dtype=torch.long)
    return torch.cat([mask_text, pad], dim=0)


def _get_input_ids_loss_mask(
    normalized_conv: list[dict],
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    *,
    # For logging
    conv_idx: int | None = None,
):
    hf_conv = _adapt_conv_for_hf(normalized_conv, processor)

    if assistant_pattern is None:
        # HF assistant token mask
        encoded_any = processor.apply_chat_template(
            hf_conv,
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
        if Version(TRANSFORMERS_VERSION) >= Version("5.4.0"):
            encoded_any = processor.apply_chat_template(
                hf_conv,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                processor_kwargs=processor_kwargs,
            )
        else:
            encoded_any = processor.apply_chat_template(
                hf_conv,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                **processor_kwargs,
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
            hf_conv,
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
        formatted_text,
        offsets,
        assistant_pattern,
        conv_idx=conv_idx,
        max_length=max_length,
    )

    return input_ids, loss_mask


def _preprocess_batch(
    examples: dict,
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    *,
    indices: list[int] | None = None,
    placeholder_token_ids: tuple[int, ...] = (),
    multimodal_output_dir: str | Path | None = None,
    sidecar_prefix: str = "dataset",
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    results: dict[str, list] = {
        "input_ids": [],
        "loss_mask": [],
        "seq_len": [],
        "messages_json": [],
        "mm_file": [],
        "use_audio_in_video": [],
    }
    include_messages = isinstance(processor, ProcessorMixin)
    if include_messages:
        results["messages"] = []
    conversations: list[dict] = examples.get("conversations", [])

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    for idx, conv in enumerate(conversations):
        sample_idx = indices[idx] if indices is not None else idx
        if not conv or not isinstance(conv, list):
            log.warning(
                f"[DROP sample_idx={sample_idx}] reason=empty_or_non_list_conversation "
                f"type={type(conv).__name__}"
            )
            continue

        normalized_conv = _normalize_conversation(conv, turn_dropout)
        if not normalized_conv:
            log.warning(
                f"[DROP sample_idx={sample_idx}] reason=normalized_conversation_empty "
                f"raw_turns={len(conv)}"
            )
            continue

        is_multimodal = include_messages and _is_multimodal_conversation(normalized_conv)
        messages = _adapt_conv_for_vllm(normalized_conv) if include_messages else []
        messages_json = _serialize_messages(messages) if is_multimodal else ""
        mm_file = ""
        use_audio_in_video = int(_conversation_use_audio_in_video(normalized_conv))

        try:
            if is_multimodal:
                allowed = _processor_kwargs(processor)
                call_kwargs: dict[str, Any] = dict(
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={},
                )
                for key in (
                    "load_audio",
                    "load_image",
                    "load_video",
                    "load_audios",
                    "load_images",
                    "load_videos",
                ):
                    if key in allowed:
                        call_kwargs[key] = True
                supports_mask = (
                    assistant_pattern is None
                    and "return_assistant_tokens_mask" in allowed
                )
                if supports_mask:
                    call_kwargs["return_assistant_tokens_mask"] = True

                encoded_any = processor.apply_chat_template(
                    _conversation_for_processor(normalized_conv), **call_kwargs
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
                    candidate_loss_mask = _maybe_strip_batch_dim(encoded[mask_key]).to(
                        torch.long
                    )
                    base_loss_mask = candidate_loss_mask if _mask_has_positive(candidate_loss_mask) else None
                else:
                    base_loss_mask = None

                if base_loss_mask is None:
                    if assistant_pattern is None:
                        assistant_pattern = _detect_assistant_pattern(processor)
                    base_loss_mask = _loss_mask_from_assistant_token_spans(
                        input_ids, normalized_conv, get_tokenizer(processor)
                    )
                    if base_loss_mask is None:
                        base_loss_mask = _loss_mask_from_ids_fallback(
                            input_ids,
                            normalized_conv,
                            get_tokenizer(processor),
                            assistant_pattern,
                            placeholder_token_ids,
                        )

                loss_mask = _build_multimodal_loss_mask(
                    input_ids, base_loss_mask, placeholder_token_ids
                )
                if len(input_ids) > max_length:
                    log.warning(
                        f"[DROP sample_idx={sample_idx}] reason=overlength_multimodal "
                        f"len(input_ids)={len(input_ids)} max_length={max_length}"
                    )
                    continue
                if multimodal_output_dir is not None:
                    mm_file = _save_multimodal_sidecar(
                        encoded, multimodal_output_dir, sample_idx, sidecar_prefix
                    )
            else:
                input_ids, loss_mask = _get_input_ids_loss_mask(
                    normalized_conv,
                    processor,
                    max_length=max_length,
                    assistant_pattern=assistant_pattern,
                    conv_idx=idx,
                )
                input_ids = torch.tensor(input_ids, dtype=torch.long)

            assert len(input_ids) == len(loss_mask), (
                f"Shape mismatch: input_ids={len(input_ids)}, loss_mask={len(loss_mask)}"
            )

            if minimum_valid_tokens is not None:
                num_valid_tokens = int(loss_mask.sum().item())
                if num_valid_tokens < minimum_valid_tokens:
                    log.warning(
                        f"[DROP sample_idx={sample_idx}] reason=too_few_valid_tokens "
                        f"num_valid_tokens={num_valid_tokens} "
                        f"minimum_valid_tokens={minimum_valid_tokens} "
                        f"len(input_ids)={len(input_ids)} is_multimodal={is_multimodal}"
                    )
                    continue

            results["input_ids"].append(input_ids)
            results["loss_mask"].append(loss_mask.to(torch.long))
            results["seq_len"].append(len(input_ids))
            if include_messages:
                results["messages"].append(messages)
            results["messages_json"].append(messages_json)
            results["mm_file"].append(mm_file)
            results["use_audio_in_video"].append(use_audio_in_video)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            log.error(
                f"[DROP sample_idx={sample_idx}] reason=exception "
                f"exc_type={type(e).__name__} "
                f"(assistant_pattern={assistant_pattern is not None}): {e}"
            )
            continue

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    processor: ProcessorLike,
    max_length: int = 2048,
    num_proc: int = 8,
    assistant_pattern: str | Pattern[str] | None = None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    placeholder_token_ids: tuple[int, ...] = (),
    multimodal_output_dir: str | Path | None = None,
    sidecar_prefix: str = "dataset",
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

    if multimodal_output_dir is not None:
        (Path(multimodal_output_dir) / MULTIMODAL_SIDECAR_DIR).mkdir(
            parents=True, exist_ok=True
        )

    original_cols = dataset.column_names

    # Avoid CPU contention for MM processing:
    # https://github.com/vllm-project/vllm/pull/31879
    with (
        set_default_torch_num_threads()
        if isinstance(processor, ProcessorMixin)
        else nullcontext()
    ):
        dataset = dataset.map(
            lambda examples, indices: _preprocess_batch(
                examples,
                processor,
                max_length,
                assistant_pattern,
                turn_dropout,
                minimum_valid_tokens,
                indices=indices,
                placeholder_token_ids=placeholder_token_ids,
                multimodal_output_dir=multimodal_output_dir,
                sidecar_prefix=sidecar_prefix,
            ),
            batched=True,
            with_indices=True,
            num_proc=num_proc,
            batch_size=400,
            remove_columns=original_cols,
            keep_in_memory=True,  # skip caching
        )

    dataset.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
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
    raw_dataset = load_dataset(config.hf_path, name=config.subset, split=config.split)

    if config.filter_fn is not None:
        raw_dataset = raw_dataset.filter(config.filter_fn)

    return raw_dataset, config.normalize_fn


def get_tokenizer(processor: ProcessorLike):
    if isinstance(processor, ProcessorMixin):
        return processor.tokenizer  # type: ignore[attr-defined]

    return processor


def _resolve_pad_token(processor: ProcessorLike):
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_processor(target_model_path: str, *, trust_remote_code: bool = False):
    processor = AutoProcessor.from_pretrained(
        target_model_path,
        trust_remote_code=trust_remote_code,
    )
    _resolve_pad_token(processor)

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
    multimodal_output_dir: Path | str | None = None,
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
    processor = load_processor(target_model_path, trust_remote_code=trust_remote_code)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        raise ValueError(
            f"Processor for {target_model_path} does not support chat templates. "
            "Please use a model with a pre-configured chat template."
        )

    placeholder_token_ids: tuple[int, ...] = ()
    verifier_config = AutoConfig.from_pretrained(
        target_model_path, trust_remote_code=trust_remote_code
    )
    multimodal_config = getattr(verifier_config, "thinker_config", verifier_config)
    if isinstance(processor, ProcessorMixin):
        placeholder_token_ids = tuple(
            int(token_id)
            for attr in ("image_token_id", "video_token_id", "audio_token_id")
            if (token_id := getattr(multimodal_config, attr, None)) is not None
        )
    if multimodal_output_dir is None:
        multimodal_output_dir = Path(token_freq_path).parent

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
            placeholder_token_ids=placeholder_token_ids,
            multimodal_output_dir=multimodal_output_dir,
            sidecar_prefix=train_data_path,
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
