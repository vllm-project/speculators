"""Convert agentic (Pi) event traces into speculator training rows.

Pi records a session as a JSONL event stream. The final ``agent_end`` event
carries the full ordered message list, which this module flattens into an
OpenAI-style conversation and feeds through the regular speculator
preprocessing pipeline (render-boundary loss masks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from speculators.data_generation.preprocessing import (
    ProcessorLike,
    build_speculator_training_dataset,
)

__all__ = ["pi_messages_to_conversation", "rows_from_pi_trace"]


def _system_content(message: dict) -> str:
    """Flatten Pi's structured system message sections into one string."""
    sections = message.get("sections")
    if isinstance(sections, dict) and sections:
        parts = [str(value) for value in sections.values() if value]
        return "\n\n".join(parts)

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _tool_to_openai(tool: dict) -> dict:
    """Wrap a Pi tool definition in the OpenAI function-tool envelope."""
    if tool.get("type") == "function" or "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters")
            or {"type": "object", "properties": {}},
        },
    }


def _block_text(content: Any) -> str:
    """Join the text blocks of a Pi content list (or pass a string through)."""
    if isinstance(content, str):
        return content
    return "".join(
        str(block.get("text", ""))
        for block in content or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def pi_messages_to_conversation(
    messages: list[dict],
) -> tuple[list[dict], list[dict] | None]:
    """Convert Pi's ordered message list to OpenAI-style conversation + tools.

    Multiple ``toolCall`` blocks in one assistant message are preserved as
    separate ``tool_calls`` entries. Thinking blocks populate both ``thinking``
    and ``reasoning_content`` so either consumer convention works.
    """
    conversation: list[dict] = []
    tools: list[dict] | None = None

    for message in messages:
        role = message.get("role")

        if role == "system":
            conversation.append({"role": "system", "content": _system_content(message)})
            added = message.get("toolsAdded")
            if added:
                tools = [_tool_to_openai(tool) for tool in added]

        elif role == "user":
            conversation.append(
                {"role": "user", "content": _block_text(message.get("content"))}
            )

        elif role == "assistant":
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict] = []

            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "thinking":
                    thinking_parts.append(str(block.get("thinking", "")))
                elif block_type == "toolCall":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("arguments") or {}),
                            },
                        }
                    )

            turn: dict = {"role": "assistant", "content": "".join(text_parts)}
            if thinking_parts:
                thinking = "".join(thinking_parts)
                turn["thinking"] = thinking
                turn["reasoning_content"] = thinking
            if tool_calls:
                turn["tool_calls"] = tool_calls
            conversation.append(turn)

        elif role == "toolResult":
            conversation.append(
                {
                    "role": "tool",
                    "content": _block_text(message.get("content")),
                    "tool_call_id": message.get("toolCallId"),
                }
            )

    return conversation, tools


def _last_agent_end_messages(trace_path: Path) -> list[dict]:
    """Return the message list from the last ``agent_end`` event in the trace."""
    messages: list[dict] | None = None
    with trace_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "agent_end":
                messages = event.get("messages") or []
    if messages is None:
        raise ValueError(f"No agent_end event found in Pi trace: {trace_path}")
    return messages


def _to_list(values: Any) -> list:
    """Normalize a dataset cell (list, tensor, or array) to a Python list."""
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _safe_decode(processor: ProcessorLike, input_ids: list[int]) -> str:
    """Decode a token sequence for review; never fail row construction."""
    try:
        decoded = processor.decode(input_ids)
    except Exception:
        return ""
    return decoded if isinstance(decoded, str) else ""


def rows_from_pi_trace(
    trace_path: str | Path,
    *,
    processor: ProcessorLike,
    render_endpoint: str,
    seq_length: int,
    primary_id: str,
) -> list[dict]:
    """Convert one Pi JSONL trace into speculator-format training rows."""
    conversation, tools = pi_messages_to_conversation(
        _last_agent_end_messages(Path(trace_path))
    )

    sample: dict[str, Any] = {"conversations": conversation}
    if tools:
        sample["tools"] = tools
    dataset = HFDataset.from_list([sample])

    built = build_speculator_training_dataset(
        dataset,
        processor,
        max_length=seq_length,
        num_proc=1,
        render_endpoint=render_endpoint,
        minimum_valid_tokens=1,
    )

    rows: list[dict] = []
    for index, row in enumerate(built):
        input_ids = _to_list(row["input_ids"])
        loss_mask = _to_list(row["loss_mask"])
        rows.append(
            {
                "id": f"{primary_id}_gen{index}",
                "primary_id": primary_id,
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "text": _safe_decode(processor, input_ids),
                "metadata": {"instance_id": primary_id, "source": "pi"},
            }
        )
    return rows
