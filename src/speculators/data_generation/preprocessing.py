import bisect
import json
import re
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from re import Pattern
from typing import Any, cast
from urllib.parse import unquote, urlparse

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from packaging.version import Version
from transformers import (
    AutoProcessor,
    BatchEncoding,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers import __version__ as TRANSFORMERS_VERSION  # noqa: N812
from transformers.image_utils import load_image

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.media import get_image_ref
from speculators.data_generation.torch_utils import set_default_torch_num_threads
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


ProcessorLike = PreTrainedTokenizerBase | ProcessorMixin


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
                image_ref = get_image_ref(item)
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


def _load_image_for_processor(image_ref: Any) -> Any:
    """Load a local/data image reference for HF multimodal processors."""
    if isinstance(image_ref, str):
        parsed = urlparse(image_ref)
        if parsed.scheme == "file":
            image_ref = unquote(parsed.path)
    return load_image(image_ref)


def _extract_processor_images_from_conversation(conv: list[dict]) -> list[Any]:
    """Extract image inputs in the same order as the chat template placeholders."""
    images = []
    for turn in conv:
        content = turn.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue
            image_ref = get_image_ref(item)
            if image_ref is not None:
                images.append(_load_image_for_processor(image_ref))

    return images


def _get_image_token_ids(
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizerBase,
) -> set[int]:
    """Return known image placeholder token IDs for VL token expansion."""
    image_token_ids: set[int] = set()

    def _add_token_id(token_id: Any) -> None:
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            image_token_ids.add(token_id)

    for source in (processor, tokenizer, getattr(processor, "tokenizer", None)):
        if source is None:
            continue
        for attr_name in ("image_token_id", "image_token_index"):
            _add_token_id(getattr(source, attr_name, None))

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert_tokens_to_ids):
        return image_token_ids

    image_tokens = {"<|image_pad|>", "<image>", "<image_pad>"}
    for source in (processor, tokenizer, getattr(processor, "tokenizer", None)):
        if source is None:
            continue
        for attr_name in ("image_token", "image_pad_token"):
            token = getattr(source, attr_name, None)
            if isinstance(token, str):
                image_tokens.add(token)

    for token in image_tokens:
        _add_token_id(convert_tokens_to_ids(token))

    return image_token_ids


def _validate_supported_multimodal_content(conv: list[dict]) -> None:
    """Fail fast for media modalities not expanded by this preprocessing path."""
    unsupported_types = {
        "audio",
        "audio_url",
        "input_audio",
        "input_video",
        "video",
        "video_url",
    }
    for turn in conv:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in unsupported_types:
                raise ValueError(
                    "Only image inputs are supported for multimodal preprocessing; "
                    f"got content part type={item.get('type')!r}"
                )


def _validate_multimodal_truncation_boundary(
    input_ids: list[int],
    max_length: int,
    image_token_ids: set[int],
) -> None:
    """Reject truncation in the middle of an expanded image-token block."""
    if max_length <= 0 or max_length >= len(input_ids):
        return
    if not image_token_ids:
        raise ValueError("Cannot validate multimodal truncation without image IDs")

    prev_token_id = input_ids[max_length - 1]
    next_token_id = input_ids[max_length]
    if prev_token_id in image_token_ids and next_token_id in image_token_ids:
        raise ValueError(
            "Refusing to truncate multimodal input in the middle of an image "
            f"token block at max_length={max_length}. Increase --seq-length "
            "or filter this sample."
        )


def _expand_loss_mask_for_multimodal_tokens(
    input_ids: list[int],
    loss_mask: torch.Tensor,
    expanded_input_ids: list[int],
    image_token_ids: set[int],
) -> torch.Tensor:
    """Expand loss masks when a processor expands image placeholders."""
    if input_ids == expanded_input_ids:
        return loss_mask

    if not image_token_ids:
        raise ValueError("Cannot align expanded multimodal input IDs without image IDs")

    mask_values = loss_mask.tolist()
    expanded_mask: list[int] = []
    src_idx = 0
    dst_idx = 0

    while src_idx < len(input_ids):
        token_id = input_ids[src_idx]
        if dst_idx >= len(expanded_input_ids):
            break

        if token_id in image_token_ids:
            start_idx = dst_idx
            while (
                dst_idx < len(expanded_input_ids)
                and expanded_input_ids[dst_idx] == token_id
            ):
                dst_idx += 1
            if dst_idx == start_idx:
                raise ValueError(
                    f"Unable to align image token {token_id} at position {src_idx}"
                )
            expanded_mask.extend([0] * (dst_idx - start_idx))
            src_idx += 1
            continue

        if expanded_input_ids[dst_idx] != token_id:
            raise ValueError(
                "Unable to align expanded multimodal input IDs at "
                f"source={src_idx}, expanded={dst_idx}: "
                f"{token_id} != {expanded_input_ids[dst_idx]}"
            )
        expanded_mask.append(int(mask_values[src_idx]))
        src_idx += 1
        dst_idx += 1

    if dst_idx != len(expanded_input_ids):
        raise ValueError(
            "Expanded multimodal input IDs contain trailing tokens after alignment"
        )

    return torch.tensor(expanded_mask, dtype=loss_mask.dtype)


def _expand_multimodal_inputs_with_images(
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizerBase,
    formatted_text: str,
    normalized_conv: list[dict],
    input_ids: list[int],
    loss_mask: torch.Tensor,
    max_length: int,
) -> tuple[list[int], torch.Tensor]:
    """Use actual images so HF preprocessing matches vLLM VL token expansion."""
    _validate_supported_multimodal_content(normalized_conv)
    images = _extract_processor_images_from_conversation(normalized_conv)
    if not images:
        return input_ids[:max_length], loss_mask[:max_length]

    encoded = processor(
        text=[formatted_text],
        images=images,
        truncation=False,
    )
    expanded_input_ids = _flatten_singleton_batch(
        encoded["input_ids"],
        field_name="Multimodal processor input_ids with images",
    )
    image_token_ids = _get_image_token_ids(processor, tokenizer)
    expanded_loss_mask = _expand_loss_mask_for_multimodal_tokens(
        input_ids,
        loss_mask,
        expanded_input_ids,
        image_token_ids,
    )
    _validate_multimodal_truncation_boundary(
        expanded_input_ids,
        max_length,
        image_token_ids,
    )
    return expanded_input_ids[:max_length], expanded_loss_mask[:max_length]


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
) -> list[dict]:
    """Normalize conversation to standard format with role/content keys.

    Args:
        conv: Raw conversation turns

    Returns:
        Normalized conversation
    """
    normalized = []
    for turn in conv:
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
            if local_path := part.get("path"):
                file_url = f"file://{Path(local_path).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}
            if url := part.get("url"):
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}
            if media_ref := part.get(modality):
                if not isinstance(media_ref, str | Path):
                    expr = {"type": modality, modality: "..."}
                    raise ValueError(
                        f"Content part {expr} is not supported. To avoid copying "
                        f"the {modality} when saving the preprocessed dataset, "
                        f"please express {modality} inputs using file paths or URLs."
                    )
                media_text = str(media_ref)
                if media_text.startswith(("http://", "https://", "data:", "file://")):
                    return {
                        "type": f"{modality}_url",
                        f"{modality}_url": {"url": media_text},
                    }
                file_url = f"file://{Path(media_text).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}

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
        mask = _flatten_singleton_batch(mask, field_name="assistant mask support check")
        return any(m == 1 for m in mask)
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


def _get_input_ids_loss_mask(
    normalized_conv: list[dict],
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    *,
    tools: list[dict] | None = None,
    # For logging
    conv_idx: int | None = None,
):
    hf_conv = _adapt_conv_for_hf(normalized_conv, processor)

    if assistant_pattern is None:
        # HF assistant token mask
        encoded_any = processor.apply_chat_template(
            hf_conv,
            tokenize=True,
            tools=tools,  # type: ignore[arg-type]
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        encoded = cast("BatchEncoding | BatchFeature", encoded_any)

        # input IDs and loss mask
        input_ids = _flatten_singleton_batch(
            encoded["input_ids"],
            field_name="apply_chat_template input_ids",
        )
        # HF uses 'assistant_masks' in recent versions
        mask_key = (
            "assistant_masks" if "assistant_masks" in encoded else "assistant_mask"
        )
        loss_mask = torch.tensor(
            _flatten_singleton_batch(
                encoded[mask_key],
                field_name="apply_chat_template assistant mask",
            ),
            dtype=torch.long,
        )

        if isinstance(processor, ProcessorMixin):
            return input_ids, loss_mask

        return input_ids[:max_length], loss_mask[:max_length]

    # Fallback: regex-based detection
    assert assistant_pattern is not None, "Assistant pattern required for fallback"

    processor_kwargs: dict = {
        "return_offsets_mapping": True,
        # Multimodal processors must expand image tokens before truncation.
        "truncation": False,
        "add_special_tokens": False,
    }

    if isinstance(processor, ProcessorMixin):
        if Version(TRANSFORMERS_VERSION) >= Version("5.4.0"):
            encoded_any = processor.apply_chat_template(
                hf_conv,
                tokenize=True,
                tools=tools,
                add_generation_prompt=False,
                return_dict=True,
                processor_kwargs=processor_kwargs,
            )
        else:
            encoded_any = processor.apply_chat_template(
                hf_conv,
                tokenize=True,
                tools=tools,
                add_generation_prompt=False,
                return_dict=True,
                **processor_kwargs,
            )

        encoded = cast("BatchFeature", encoded_any)

        input_ids = _flatten_singleton_batch(
            encoded["input_ids"],
            field_name="processor apply_chat_template input_ids",
        )
        offsets = _flatten_singleton_batch(
            encoded["offset_mapping"],
            field_name="processor apply_chat_template offset_mapping",
        )

        # MM placeholder tokens are inserted separate from chat template
        formatted_text = processor.decode(input_ids)
        assert isinstance(formatted_text, str)
    else:
        processor_kwargs["max_length"] = max_length
        processor_kwargs["truncation"] = True

        # More optimized flow for text-only processors (i.e. tokenizers)
        formatted_text = processor.apply_chat_template(
            hf_conv,
            tokenize=False,
            tools=tools,  # type: ignore[arg-type]
            add_generation_prompt=False,
        )
        assert isinstance(formatted_text, str)

        # Tokenize and get offsets
        encoded_any = processor(formatted_text, **processor_kwargs)
        encoded = cast("BatchEncoding", encoded_any)

        input_ids = _flatten_singleton_batch(
            encoded["input_ids"], field_name="tokenizer input_ids"
        )
        offsets = _flatten_singleton_batch(
            encoded["offset_mapping"], field_name="tokenizer offset_mapping"
        )

    loss_mask = _create_loss_mask_from_offsets(
        formatted_text,
        offsets,
        assistant_pattern,
        conv_idx=conv_idx,
        max_length=max_length,
    )

    return input_ids, loss_mask


def _parse_conv_tools(conv_tools: object, idx: int) -> list | None:
    """Parse the tools JSON string for one conversation; warn and return None
    on invalid JSON or unexpected types."""
    if not conv_tools:
        return None
    if isinstance(conv_tools, list):
        return conv_tools
    if not isinstance(conv_tools, str):
        log.warning(
            f"Non-string value in tools column for conversation {idx}: "
            f"{type(conv_tools).__name__}, proceeding without tools"
        )
        return None
    try:
        return json.loads(conv_tools)
    except json.JSONDecodeError as e:
        log.warning(
            f"Invalid JSON in tools column for conversation {idx}: {e}, "
            "proceeding without tools"
        )
        return None


def _passthrough_pretokenized(
    examples: dict, max_length: int, minimum_valid_tokens: int | None = None
) -> dict[str, list]:
    """Carry pre-tokenized ``(input_ids, loss_mask)`` rows through, truncated only.

    On-policy regeneration already applied the boundary as the mask, so these rows
    need no chat-template rendering or regex span detection.
    """
    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    for ids, mask in zip(examples["input_ids"], examples["loss_mask"], strict=True):
        # `strict=True` only pairs the columns; a per-row skew would survive it and
        # the collator packs each key independently, silently shifting the mask
        # against the ids for every sample packed after this one.
        if len(ids) != len(mask):
            raise ValueError(
                f"Pre-tokenized row shape mismatch: "
                f"input_ids={len(ids)}, loss_mask={len(mask)}"
            )
        trimmed_ids = ids[:max_length]
        trimmed_mask = mask[:max_length]
        if (
            minimum_valid_tokens is not None
            and sum(trimmed_mask) < minimum_valid_tokens
        ):
            continue
        results["input_ids"].append(torch.tensor(trimmed_ids, dtype=torch.long))
        results["loss_mask"].append(torch.tensor(trimmed_mask, dtype=torch.long))
        results["seq_len"].append(len(trimmed_ids))
    return results


def _preprocess_batch(
    examples: dict,
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    minimum_valid_tokens: int | None = None,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks."""

    # On-policy regeneration rows are already masked (boundary); pass them through
    # instead of re-tokenizing and re-masking.
    if "input_ids" in examples and "loss_mask" in examples:
        return _passthrough_pretokenized(examples, max_length, minimum_valid_tokens)

    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    conversations = _get_conversations_from_examples(examples)

    # MM inputs must use Chat Completions API
    if isinstance(processor, ProcessorMixin):
        results["messages"] = []

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    tools_col = examples.get("tools")
    if tools_col is not None and len(tools_col) != len(conversations):
        log.warning(
            f"Tools column length ({len(tools_col)}) does not match "
            f"conversations length ({len(conversations)}), proceeding without tools"
        )
        tools_col = None

    for idx, conv in enumerate(conversations):
        conv_tools = tools_col[idx] if tools_col is not None else None

        if not conv or not isinstance(conv, list):
            continue

        # Normalize to standard format
        normalized_conv = _normalize_conversation(conv)
        if not normalized_conv:
            continue

        parsed_tools = _parse_conv_tools(conv_tools, idx)

        try:
            formatted_raw = None
            if isinstance(processor, ProcessorMixin):
                formatted_raw_any = processor.apply_chat_template(
                    _adapt_conv_for_hf(normalized_conv, processor),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                assert isinstance(formatted_raw_any, str)
                formatted_raw = formatted_raw_any

            input_ids, loss_mask = _get_input_ids_loss_mask(
                normalized_conv,
                processor,
                max_length=max_length,
                assistant_pattern=assistant_pattern,
                tools=parsed_tools,
                conv_idx=idx,
            )
            if isinstance(processor, ProcessorMixin):
                input_ids, loss_mask = _expand_multimodal_inputs_with_images(
                    processor,
                    get_tokenizer(processor),
                    formatted_raw or "",
                    normalized_conv,
                    list(input_ids),
                    loss_mask,
                    max_length,
                )
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
            OSError,
        ) as e:
            raise RuntimeError(
                f"Failed to process conversation {idx} "
                f"(assistant_pattern={assistant_pattern is not None}): {e}"
            ) from e

        # Assert shapes match
        assert len(input_ids) == len(loss_mask), (
            f"Shape mismatch: input_ids={len(input_ids)}, loss_mask={len(loss_mask)}"
        )

        # Bound both to max_length: a turn running past the window keeps only its
        # in-window tokens, and input_ids/loss_mask stay aligned and bounded.
        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]

        # Filtering samples out with too few valid tokens
        if minimum_valid_tokens is not None:
            num_valid_tokens = int(loss_mask.sum().item())
            if num_valid_tokens < minimum_valid_tokens:
                continue

        # Append to results
        results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
        results["loss_mask"].append(loss_mask)
        results["seq_len"].append(len(input_ids))

        if "messages" in results:
            results["messages"].append(_adapt_conv_for_vllm(normalized_conv))

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    processor: ProcessorLike,
    max_length: int = 2048,
    num_proc: int = 8,
    assistant_pattern: str | Pattern[str] | None = None,
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
        minimum_valid_tokens: Number of tokens to consider for a valid sample
    """
    original_cols = dataset.column_names
    # These rows carry the generation boundary as their mask, so _preprocess_batch
    # passes them through: no chat template, no span detection.
    pretokenized = {"input_ids", "loss_mask"} <= set(original_cols)

    if pretokenized:
        log.info("Pre-tokenized rows: using their loss mask, skipping chat template")
        if assistant_pattern is not None:
            log.warning(
                "assistant_pattern does not apply to pre-tokenized rows; ignoring"
            )
    # Detect and use provided assistant message pattern
    elif assistant_pattern is not None:
        log.info(f"Using custom assistant pattern: {str(assistant_pattern)[:80]}...")
    elif _supports_assistant_mask(processor):
        assistant_pattern = None  # Signal to use HF mask in _preprocess_batch
        log.info("Using HF assistant token mask for loss masking")
    else:
        assistant_pattern = _detect_assistant_pattern(processor)
        log.info(f"Detected assistant pattern: {str(assistant_pattern)[:80]}...")

    remove_cols = original_cols
    if isinstance(processor, ProcessorMixin):
        remove_cols = [col for col in original_cols if col != "messages"]

    # Avoid CPU contention for MM processing:
    # https://github.com/vllm-project/vllm/pull/31879
    with (
        set_default_torch_num_threads()
        if isinstance(processor, ProcessorMixin)
        else nullcontext()
    ):
        dataset = dataset.map(
            lambda examples: _preprocess_batch(
                examples,
                processor,
                max_length,
                assistant_pattern,
                minimum_valid_tokens,
            ),
            batched=True,
            num_proc=num_proc,
            # Multimodal preprocessing loads each image and asks the processor to
            # expand image placeholders into the real visual-token length. Large
            # batches can look stuck because a worker must finish the whole batch
            # before datasets updates progress.
            batch_size=32 if isinstance(processor, ProcessorMixin) else 1000,
            remove_columns=remove_cols,
            keep_in_memory=True,  # skip caching
        )

    if isinstance(processor, ProcessorMixin):
        dataset.set_format(
            type="torch", columns=["input_ids", "loss_mask"], output_all_columns=True
        )
    else:
        dataset.set_format(type="torch")
    return dataset


def _load_hf_dataset(spec: str) -> tuple[HFDataset, None]:
    """Load an arbitrary HuggingFace dataset from an ``hf:`` spec.

    Args:
        spec: ``hf:<dataset_id>[:<subset>:<split>]``. The split defaults to
            ``train``. A single suffix (``hf:<id>:<split>``) selects a split
            without a subset; both can be given as ``hf:<id>:<subset>:<split>``.

    Returns:
        Tuple of (raw_dataset, None). No normalize_fn is applied: the dataset
        must already be in conversations format.

    Raises:
        ValueError: If the spec is malformed or the loaded dataset has no
            ``conversations`` column.
    """
    subset: str | None
    match spec.removeprefix("hf:").split(":"):
        case [hf_id]:
            subset, split = None, "train"
        case [hf_id, split]:
            subset = None
        case [hf_id, subset, split]:
            pass
        case _:
            raise ValueError(
                f"Invalid hf: spec '{spec}'. "
                f"Expected hf:<dataset_id>[:<subset>:<split>]."
            )

    if not hf_id:
        raise ValueError(f"Invalid hf: spec '{spec}': missing dataset id.")
    if subset == "":
        raise ValueError(f"Invalid hf: spec '{spec}': empty subset.")
    if not split:
        raise ValueError(f"Invalid hf: spec '{spec}': empty split.")

    raw_dataset = load_dataset(hf_id, name=subset, split=split)

    if "conversations" not in raw_dataset.column_names:
        raise ValueError(
            f"HuggingFace dataset '{hf_id}' (split '{split}') is not in "
            f"conversations format: expected a 'conversations' column but found "
            f"{raw_dataset.column_names}. Pass a dataset already in conversations "
            f"format, or add a preset to DATASET_CONFIGS with a normalize_fn."
        )

    return raw_dataset, None


def load_raw_dataset(
    train_data_path: str,
) -> tuple[HFDataset, Callable[[dict], dict] | None]:
    """Load a raw dataset from one of several source types.

    Resolution order:
        1. Local ``.json``/``.jsonl`` file.
        2. Local directory: recursively load all ``*.json``/``*.jsonl`` files
           as a single dataset.
        3. Named preset from ``DATASET_CONFIGS``.
        4. ``hf:<id>[:<subset>:<split>]`` for an arbitrary HuggingFace dataset.

    Args:
        train_data_path: File path, directory path, preset name, or ``hf:`` spec.

    Returns:
        Tuple of (raw_dataset, normalize_fn). normalize_fn is None for sources
        already in conversations format.

    Raises:
        ValueError: If the source cannot be resolved or a local directory
            contains no ``.json``/``.jsonl`` files.
    """
    # 1. Local file
    if train_data_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=train_data_path, split="train"), None

    # 2. Local directory
    path = Path(train_data_path)
    if path.is_dir():
        data_files = sorted(
            str(p) for p in (*path.rglob("*.json"), *path.rglob("*.jsonl"))
        )
        if not data_files:
            raise ValueError(
                f"No .json/.jsonl files found in directory: {train_data_path}"
            )
        return load_dataset("json", data_files=data_files, split="train"), None

    # 3. Named preset
    if train_data_path in DATASET_CONFIGS:
        config = DATASET_CONFIGS[train_data_path]
        raw_dataset = load_dataset(
            config.hf_path, name=config.subset, split=config.split
        )
        if config.filter_fn is not None:
            raw_dataset = raw_dataset.filter(config.filter_fn)
        return raw_dataset, config.normalize_fn

    # 4. Arbitrary HuggingFace dataset
    if train_data_path.startswith("hf:"):
        return _load_hf_dataset(train_data_path)

    raise ValueError(
        f"Unsupported dataset: {train_data_path}. Supported: local .json/.jsonl "
        f"file, local directory of .json/.jsonl files, hf:<id>[:<subset>:<split>], "
        f"or a preset {list(DATASET_CONFIGS.keys())}."
    )


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
    minimum_valid_tokens: int | None = None,
    allow_empty_output: bool = False,
    trust_remote_code: bool = False,
    is_multimodal: bool | None = None,
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
        minimum_valid_tokens: Number of tokens to consider for a valid sample
        allow_empty_output: If True, allow returning an empty dataset instead of
                          raising when no samples survive preprocessing.
        trust_remote_code: If True, allows executing code from HF Hub.
        is_multimodal: Optional compatibility flag. When True, require a
                       ProcessorMixin-backed processor. When None, detect
                       multimodal preprocessing automatically.

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
    processor_is_multimodal = isinstance(processor, ProcessorMixin)
    if is_multimodal and not processor_is_multimodal:
        raise ValueError(
            f"Processor for {target_model_path} is not a multimodal ProcessorMixin."
        )
    log.info(f"Using multimodal mode: {processor_is_multimodal}")

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

        preprocessed_dataset = build_eagle3_dataset(
            dataset=raw_dataset,
            processor=processor,
            max_length=seq_length,
            num_proc=build_dataset_num_proc,
            assistant_pattern=assistant_pattern,
            minimum_valid_tokens=minimum_valid_tokens,
        )
        if minimum_valid_tokens is not None:
            log.info(f"Kept {len(preprocessed_dataset)} samples after filtering")
        processed_datasets.append(preprocessed_dataset)

    combined_dataset = concatenate_datasets(processed_datasets)
    combined_dataset = combined_dataset.shuffle(seed=seed)
    if max_samples is not None and len(combined_dataset) > max_samples:
        combined_dataset = combined_dataset.select(range(max_samples))

    if len(combined_dataset) == 0 and not allow_empty_output:
        raise ValueError(
            "No samples remain after preprocessing. Check the dataset schema, "
            "assistant masking, and --minimum-valid-tokens. Pass "
            "--allow-empty-output if an empty dataset is intentional."
        )

    log.subsection("Computing token frequency distribution")
    save_token_frequency_distribution(
        dataset=combined_dataset,
        output_path=token_freq_path,
    )

    if len(combined_dataset) == 0:
        log.warning("No samples remain after preprocessing; skipping visualization")
    else:
        log.subsection("Visualizing sample")
        _visualize_sample(combined_dataset, processor, idx=0)

    log.section("Dataset preprocessing complete")

    return combined_dataset, processor
