import bisect
import inspect
import json
import random
import re
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from re import Pattern
from typing import cast

from jinja2.exceptions import TemplateError
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

ProcessorLike = PreTrainedTokenizerBase | ProcessorMixin


def _visualize_sample(preprocessed: HFDataset, processor: ProcessorLike, idx: int = 0):
    """Visualize a single sample with color-coded trainable regions."""
    prep_sample = preprocessed[idx]
    input_ids = prep_sample["input_ids"].tolist()
    loss_mask = prep_sample["loss_mask"].tolist()

    log.info(f"SAMPLE #{idx}")
    log.info("HIGHLIGHTED TEXT (BLUE = trainable, GREY = masked)")

    blue = "\033[38;5;153m"
    grey = "\033[90m"
    reset = "\033[0m"

    output = []
    prev_state = None

    for i in range(len(input_ids)):
        is_train = loss_mask[i] == 1
        token = processor.decode([input_ids[i]])
        assert isinstance(token, str)

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
    """Build loss mask for multimodal samples by zeroing out placeholder positions."""
    loss_mask = base_loss_mask.to(dtype=torch.long).clone()
    valid_ids = [tid for tid in placeholder_token_ids if tid >= 0]
    if not valid_ids:
        return loss_mask
    placeholder_tensor = torch.as_tensor(
        valid_ids, dtype=input_ids.dtype, device=input_ids.device
    )
    is_placeholder = torch.isin(input_ids, placeholder_tensor)
    loss_mask.masked_fill_(is_placeholder, 0)
    return loss_mask


def _mask_has_positive(mask: Any) -> bool:
    """Return True only when an assistant mask contains at least one trainable token."""
    mask_tensor = _maybe_strip_batch_dim(mask)
    return bool(mask_tensor.numel() > 0 and torch.count_nonzero(mask_tensor).item() > 0)


_PROCESSOR_KW_CACHE: dict[int, set[str]] = {}


def _processor_kwargs(processor: Any) -> set[str]:
    """Discover which kwargs ``processor.apply_chat_template`` explicitly accepts."""
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
    """Convert message content to HF processor-compatible content blocks."""
    if isinstance(content, list):
        return [
            seg if isinstance(seg, dict) else {"type": "text", "text": str(seg)}
            for seg in content
        ]
    return [{"type": "text", "text": content if isinstance(content, str) else str(content or "")}]


def _conversation_for_processor(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow processor-only copy with list/dict content blocks."""
    return [
        {
            **turn,
            "content": _as_processor_content_blocks(turn.get("content", "")),
        }
        for turn in conv
    ]


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
    """Normalize conversation to standard format with role/content keys."""
    num_turns_to_keep = random.randint(1, len(conv)) if turn_dropout else len(conv)

    normalized = []
    for i, turn in enumerate(conv):
        role = turn.get("from", turn.get("role", ""))
        content = _normalize_turn_content(turn.get("value") or turn.get("content") or "")

        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            log.warning(f"Unknown role '{role}', skipping turn")
            continue

        normalized_turn = {"role": role, "content": content}

        if "thinking" in turn and turn["thinking"]:
            normalized_turn["thinking"] = turn["thinking"]

        normalized.append(normalized_turn)

        if i + 1 >= num_turns_to_keep and role == "assistant":
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
            if local_path := part.get("path"):
                file_url = f"file://{Path(local_path).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}
            if url := part.get("url"):
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}

            if part.get("base64"):
                expr = {"type": modality, "base64": "..."}
                raise ValueError(
                    f"Content part {expr} is not supported. To avoid copying "
                    f"the {modality} when saving the preprocessed dataset, "
                    f"please express {modality} inputs using file paths or URLs."
                )
            if part.get(modality):
                expr = {"type": modality, modality: "..."}
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

    Merged from HEAD and incoming:
    - Uses processor.apply_chat_template (incoming)
    - Uses _mask_has_positive for validation (HEAD)
    - Adds version check for robustness
    """
    test_conv = _adapt_conv_for_hf(
        [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "test"},
        ],
        processor,
    )

    try:
        # Try with processor.apply_chat_template (incoming approach)
        res_any = processor.apply_chat_template(
            test_conv,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        res = cast("BatchEncoding | BatchFeature", res_any)

        # Check both singular and plural key names
        mask = res.get("assistant_masks", res.get("assistant_mask"))
        if mask is None:
            return False

        # Verify the mask is not all zeros (HEAD's _mask_has_positive logic)
        return _mask_has_positive(mask)

    except (TypeError, ValueError, KeyError, AttributeError, TemplateError) as e:
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

    # Strip <think>...</think> blocks from the role marker
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
    conv_idx: int | None = None,
    max_length: int | None = None,
) -> torch.Tensor:
    """Create loss mask by finding assistant response spans in formatted text."""
    loss_mask = torch.zeros(len(offsets), dtype=torch.bool)

    matches_found = 0
    token_starts = [offset[0] for offset in offsets]

    for match in re.finditer(assistant_pattern, text, re.DOTALL):
        matches_found += 1

        span_start_char = match.start(1)
        span_end_char = match.end(1)

        start_idx = bisect.bisect_left(token_starts, span_start_char)

        for idx in range(max(0, start_idx - 1), len(offsets)):
            token_start, token_end = offsets[idx]
            if token_start >= span_end_char:
                break
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


def _align_mask_to_processor_ids(
    loss_mask: torch.Tensor,
    processor_input_ids: torch.Tensor,
    placeholder_token_ids: tuple[int, ...],
) -> torch.Tensor:
    """Align tokenizer-based loss mask to processor input_ids with placeholders.

    Key function from HEAD's _loss_mask_from_ids_fallback.
    Handles the positional shift introduced by multimodal placeholder tokens.
    """
    if not placeholder_token_ids:
        return loss_mask

    target_len = int(processor_input_ids.shape[0])
    placeholder_tensor = torch.as_tensor(
        placeholder_token_ids, dtype=processor_input_ids.dtype, device=processor_input_ids.device
    )

    is_placeholder = torch.isin(processor_input_ids, placeholder_tensor)

    if not bool(is_placeholder.any()):
        # No placeholders found, return as-is (with length adjustment)
        if loss_mask.shape[0] == target_len:
            return loss_mask
        if loss_mask.shape[0] > target_len:
            return loss_mask[:target_len]
        pad = torch.zeros(target_len - loss_mask.shape[0], dtype=loss_mask.dtype)
        return torch.cat([loss_mask, pad], dim=0)

    # Align: insert 0s at placeholder positions
    aligned = torch.zeros(target_len, dtype=loss_mask.dtype)
    text_positions = (~is_placeholder).nonzero(as_tuple=True)[0]
    copy_len = min(int(text_positions.shape[0]), int(loss_mask.shape[0]))
    if copy_len > 0:
        aligned.index_copy_(0, text_positions[:copy_len], loss_mask[:copy_len])

    return aligned


def _get_input_ids_loss_mask(
    normalized_conv: list[dict],
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    placeholder_token_ids: tuple[int, ...] = (),
    *,
    conv_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified function to get input_ids and loss_mask.

    Merged version:
    - Uses incoming's clean interface
    - Integrates HEAD's placeholder alignment logic
    - Handles both HF assistant mask and regex fallback
    """
    hf_conv = _adapt_conv_for_hf(normalized_conv, processor)

    # Path 1: Use HF assistant token mask (if supported)
    if assistant_pattern is None:
        try:
            encoded_any = processor.apply_chat_template(
                hf_conv,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            encoded = cast("BatchEncoding | BatchFeature", encoded_any)

            input_ids = encoded["input_ids"]
            mask_key = (
                "assistant_masks" if "assistant_masks" in encoded else "assistant_mask"
            )

            if mask_key in encoded and _mask_has_positive(encoded[mask_key]):
                loss_mask = torch.tensor(encoded[mask_key], dtype=torch.long)
                return input_ids, loss_mask

        except (TypeError, ValueError, KeyError, AttributeError, TemplateError):
            # Fall through to regex fallback
            pass

    # Path 2: Regex-based detection (fallback)
    assert assistant_pattern is not None, "Assistant pattern required for fallback"

    processor_kwargs: dict = {
        "return_offsets_mapping": True,
        "max_length": max_length,
        "truncation": True,
        "add_special_tokens": False,
    }

    # Handle different transformer versions (from incoming)
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

        (input_ids,) = encoded["input_ids"]
        (offsets,) = encoded["offset_mapping"]

        formatted_text = processor.decode(input_ids)
        assert isinstance(formatted_text, str)
    else:
        # Text-only processors (tokenizers)
        formatted_text = processor.apply_chat_template(
            hf_conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(formatted_text, str)

        encoded_any = processor(formatted_text, **processor_kwargs)
        encoded = cast("BatchEncoding", encoded_any)

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

    # Create loss mask from offsets
    loss_mask = _create_loss_mask_from_offsets(
        formatted_text,
        offsets,
        assistant_pattern,
        conv_idx=conv_idx,
        max_length=max_length,
    ).to(torch.long)

    # KEY: Align mask to processor input_ids if placeholders exist (from HEAD)
    if isinstance(processor, ProcessorMixin) and placeholder_token_ids:
        loss_mask = _align_mask_to_processor_ids(
            loss_mask, input_ids, placeholder_token_ids
        )

        # Also apply _build_multimodal_loss_mask logic
        loss_mask = _build_multimodal_loss_mask(
            input_ids, loss_mask, placeholder_token_ids
        )

    return input_ids, loss_mask


def _preprocess_batch(
    examples: dict,
    processor: ProcessorLike,
    max_length: int,
    assistant_pattern: str | Pattern[str] | None,
    turn_dropout: bool = False,
    minimum_valid_tokens: int | None = None,
    placeholder_token_ids: tuple[int, ...] = (),
    multimodal_output_dir: str | Path | None = None,
    sidecar_prefix: str = "dataset",
) -> dict[str, list]:
    """Process a batch of conversations into tokenized format with loss masks.

    Merged version:
    - Uses incoming's clean interface (processor-only)
    - Integrates HEAD's multimodal sidecar and messages_json
    - Handles placeholder alignment via _get_input_ids_loss_mask
    """
    results: dict[str, list] = {
        "input_ids": [],
        "loss_mask": [],
        "seq_len": [],
    }

    # Multimodal outputs (from HEAD)
    if isinstance(processor, ProcessorMixin):
        results["mm_file"] = []
        results["messages_json"] = []
        results["use_audio_in_video"] = []
        results["messages"] = []  # For vLLM Chat Completions API

    conversations: list[dict] = examples.get("conversations", [])

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    for idx, conv in enumerate(conversations):
        sample_idx = idx  # Use batch index as sample index

        if not conv or not isinstance(conv, list):
            log.warning(
                f"[DROP sample_idx={sample_idx}] reason=empty_or_non_list_conversation "
                f"type={type(conv).__name__}"
            )
            continue

        # Normalize to standard format with optional turn dropout
        normalized_conv = _normalize_conversation(conv, turn_dropout)
        if not normalized_conv:
            log.warning(
                f"[DROP sample_idx={sample_idx}] reason=normalized_conversation_empty "
                f"raw_turns={len(conv)}"
            )
            continue

        is_multimodal = _is_multimodal_batch({"conversations": [normalized_conv]})
        messages_json = _serialize_messages(normalized_conv) if is_multimodal else ""
        mm_file = ""
        use_audio_in_video = int(_conversation_use_audio_in_video(normalized_conv))

        try:
            # Use unified _get_input_ids_loss_mask (from incoming, with HEAD's alignment)
            input_ids, loss_mask = _get_input_ids_loss_mask(
                normalized_conv,
                processor,
                max_length=max_length,
                assistant_pattern=assistant_pattern,
                placeholder_token_ids=placeholder_token_ids,
                conv_idx=idx,
            )

            # Assert shapes match
            assert len(input_ids) == len(loss_mask), (
                f"Shape mismatch: input_ids={len(input_ids)}, "
                f"loss_mask={len(loss_mask)}"
            )

            # Multimodal processor tokenization must stay untruncated
            if is_multimodal and len(input_ids) > max_length:
                log.warning(
                    f"[DROP sample_idx={sample_idx}] reason=overlength_multimodal "
                    f"len(input_ids)={len(input_ids)} max_length={max_length}"
                )
                continue

            # Filter samples with too few valid tokens
            if minimum_valid_tokens is not None:
                num_valid_tokens = int(loss_mask.sum().item())
                if num_valid_tokens < minimum_valid_tokens:
                    log.warning(
                        f"[DROP sample_idx={sample_idx}] reason=too_few_valid_tokens "
                        f"num_valid_tokens={num_valid_tokens} "
                        f"minimum_valid_tokens={minimum_valid_tokens} "
                        f"len(input_ids)={len(input_ids)} "
                        f"is_multimodal={is_multimodal}"
                    )
                    continue

            # Save multimodal sidecar (from HEAD)
            if is_multimodal and multimodal_output_dir is not None:
                # Need encoded dict for sidecar
                encoded_any = processor.apply_chat_template(
                    _conversation_for_processor(normalized_conv, processor),
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                )
                mm_file = _save_multimodal_sidecar(
                    encoded_any,
                    multimodal_output_dir,
                    sample_idx,
                    sidecar_prefix,
                )

            # Append to results
            results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
            results["loss_mask"].append(loss_mask)
            results["seq_len"].append(len(input_ids))

            if "mm_file" in results:
                results["mm_file"].append(mm_file)
                results["messages_json"].append(messages_json)
                results["use_audio_in_video"].append(use_audio_in_video)

        if "messages" in results:
            results["messages"].append(_adapt_conv_for_vllm(normalized_conv))

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

    Merged version:
    - Uses incoming's clean interface with set_default_torch_num_threads
    - Keeps multimodal support from HEAD
    - Uses unified _preprocess_batch
    """
    # Detect and use provided assistant message pattern
    if assistant_pattern is not None:
        log.info(f"Using custom assistant pattern: {str(assistant_pattern)[:80]}...")
    elif _supports_assistant_mask(processor):
        assistant_pattern = None  # Signal to use HF mask
        log.info("Using HF assistant token mask for loss masking")
    else:
        assistant_pattern = _detect_assistant_pattern(processor)
        log.info(f"Detected assistant pattern: {str(assistant_pattern)[:80]}...")

    if multimodal_output_dir is not None:
        (Path(multimodal_output_dir) / MULTIMODAL_SIDECAR_DIR).mkdir(
            parents=True, exist_ok=True
        )

    original_cols = dataset.column_names

    # Avoid CPU contention for MM processing (from incoming)
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
                turn_dropout,
                minimum_valid_tokens,
                placeholder_token_ids=placeholder_token_ids,
                multimodal_output_dir=multimodal_output_dir,
                sidecar_prefix=sidecar_prefix,
            ),
            batched=True,
            num_proc=num_proc,
            batch_size=1000,
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
    """Extract tokenizer from processor if it's a ProcessorMixin."""
    if isinstance(processor, ProcessorMixin):
        return processor.tokenizer  # type: ignore[attr-defined]

    return processor


def _resolve_pad_token(processor: ProcessorLike):
    """Ensure processor has a pad token."""
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_processor(target_model_path: str, *, trust_remote_code: bool = False):
    """Load processor from pretrained model."""
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
    multimodal: bool = False,
    multimodal_output_dir: Path | str | None = None,
) -> tuple[HFDataset, ProcessorLike]:
    """Load, tokenize, and preprocess a dataset for EAGLE3 training.

    Merged version:
    - Uses incoming's function signature (trust_remote_code parameter)
    - Keeps HEAD's multimodal support
    - Unified processor handling
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

    log.subsection("Loading processor")
    processor = load_processor(target_model_path, trust_remote_code=trust_remote_code)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        raise ValueError(
            f"Processor for {target_model_path} does not support chat templates. "
            "Please use a model with a pre-configured chat template."
        )

    # Setup multimodal support (from HEAD)
    placeholder_token_ids: tuple[int, ...] = ()
    if multimodal:
        verifier_config = AutoConfig.from_pretrained(
            target_model_path, trust_remote_code=True
        )
        multimodal_config = getattr(verifier_config, "thinker_config", verifier_config)
        placeholder_token_ids = tuple(
            int(token_id)
            for attr in ("image_token_id", "video_token_id", "audio_token_id")
            if (token_id := getattr(multimodal_config, attr, None)) is not None
        )

        # Ensure multimodal output directory exists
        if multimodal_output_dir is not None:
            Path(multimodal_output_dir).mkdir(parents=True, exist_ok=True)

    processed_datasets = []
    for train_data_path in train_data_paths:
        log.subsection(f"Processing {train_data_path}")
        raw_dataset, normalize_fn = load_raw_dataset(train_data_path)
        raw_dataset = raw_dataset.shuffle(seed=seed)

        if max_samples is not None and len(raw_dataset) > 3 * max_samples:
            raw_dataset = raw_dataset.select(range(3 * max_samples))

        if normalize_fn is not None:
            raw_dataset = raw_dataset.map(
                normalize_fn,
                num_proc=build_dataset_num_proc,
                keep_in_memory=True,
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
    combined_dataset = combined_dataset.shuffle(seed=seed)

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