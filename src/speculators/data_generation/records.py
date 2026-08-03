"""Canonical records shared by response generation and data preparation."""

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class _RequiredPreparedSample(TypedDict):
    """The only representation consumed by speculator training."""

    input_ids: list[int]
    loss_mask: list[int]


class PreparedSample(_RequiredPreparedSample, total=False):
    """A prepared sample with optional source messages for multimodal serving."""

    messages: list[dict[str, Any]]


def prepared_sample(
    input_ids: list[int],
    boundary: int,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> PreparedSample:
    """Build a sample whose tokens at and after ``boundary`` are supervised."""
    if not 0 <= boundary <= len(input_ids):
        raise ValueError(
            f"Boundary {boundary} is outside a sequence of length {len(input_ids)}"
        )
    sample: PreparedSample = {
        "input_ids": input_ids,
        "loss_mask": [0] * boundary + [1] * (len(input_ids) - boundary),
    }
    if messages is not None:
        sample["messages"] = messages
    return sample


_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def normalize_conversation(conversation: object) -> list[dict[str, Any]]:
    """Normalize role/content and from/value messages to one representation."""
    if not isinstance(conversation, list):
        return []

    normalized: list[dict[str, Any]] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue

        raw_role = turn.get("role") or turn.get("from")
        role = _ROLE_ALIASES.get(raw_role) if isinstance(raw_role, str) else None
        if role is None:
            logger.warning("Skipping conversation turn with unknown role %r", raw_role)
            continue

        content = turn.get("content")
        if content is None:
            content = turn.get("value")
        message: dict[str, Any] = {"role": role, "content": content or ""}

        for key in ("tool_calls", "tool_call_id", "name"):
            if turn.get(key):
                message[key] = turn[key]

        reasoning = turn.get("reasoning_content") or turn.get("thinking")
        if reasoning:
            # vLLM consumes reasoning_content. Keep thinking for source-schema
            # round-tripping until all callers use the canonical key.
            message["reasoning_content"] = reasoning
            message["thinking"] = reasoning

        normalized.append(message)
    return normalized


def conversation_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and normalize the messages/conversations field from one row."""
    conversation = row.get("messages")
    if not (isinstance(conversation, list) and conversation):
        conversation = row.get("conversations")
    return normalize_conversation(conversation)


def parse_tools(value: object) -> list[dict[str, Any]] | None:
    """Parse an optional OpenAI-style tool list, rejecting malformed schemas."""
    if value is None or value in ("", [], {}):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tools is not valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("tools must be a list or a JSON-encoded list")
    if not all(isinstance(tool, dict) for tool in value):
        raise ValueError("every tool must be an object")
    return value or None


def _adapt_content_part(part: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]
    if part_type == "text":
        return {"type": "text", "text": part["text"]}

    if part_type in ("image", "video", "audio"):
        local_path = part.get("path")
        remote_url = part.get("url")
        if isinstance(local_path, str) and local_path:
            url = f"file://{Path(local_path).absolute()}"
        elif isinstance(remote_url, str) and remote_url:
            url = remote_url
        elif part.get("base64") or part.get(part_type):
            raise ValueError(
                f"Inline {part_type} data is not supported; use a path or URL"
            )
        else:
            raise ValueError(f"{part_type} content requires a path or URL")
        return {"type": f"{part_type}_url", f"{part_type}_url": {"url": url}}

    raise ValueError(f"Unsupported content type: {part_type!r}")


def adapt_conversation_for_vllm(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert canonical multimodal content parts to the vLLM API schema."""
    return [
        message
        if isinstance(message["content"], str)
        else message
        | {"content": [_adapt_content_part(part) for part in message["content"]]}
        for message in conversation
    ]
