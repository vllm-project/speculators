"""Convert agentic (Pi) event traces into speculator training rows.

Pi records a session as a JSONL event stream. Each ``agent_end`` event closes
one low-level run; retries, compaction, and follow-up work can start more runs.
This module concatenates the messages from completed runs into one ordered
conversation and runs it through the regular speculator preprocessing pipeline,
which applies render-boundary loss masks.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from speculators.data_generation.preprocessing import (
    ProcessorLike,
    _normalize_conversation,
    _render_boundary_rows,
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
    """Join the text blocks of a Pi content list, or pass a string through."""
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
    """Convert Pi's ordered message list to an OpenAI-style conversation and tools.

    The converter keeps multiple ``toolCall`` blocks in one assistant message as
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


def _completed_run_messages(trace_path: Path) -> list[dict]:
    """Concatenate messages from every ``agent_end`` run.

    Each ``agent_end`` carries only the messages produced by that low-level run,
    so retries, compaction, and follow-up work append to the earlier runs. A run
    with ``willRetry`` ends in the failed assistant turn that the retry replaces,
    so that turn is dropped and the rest of the run is kept. Repeated system and
    user messages are dropped so a re-emitted prompt does not appear twice.
    """
    messages: list[dict] = []
    seen_context: set[tuple[str, str]] = set()
    found_run = False

    with trace_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "agent_end":
                continue
            found_run = True
            run_messages = event.get("messages") or []
            if (
                event.get("willRetry")
                and run_messages
                and run_messages[-1].get("role") == "assistant"
            ):
                run_messages = run_messages[:-1]
            for message in run_messages:
                role = message.get("role")
                if role in ("system", "user"):
                    key = (role, json.dumps(message, sort_keys=True))
                    if key in seen_context:
                        continue
                    seen_context.add(key)
                messages.append(message)

    if not found_run:
        raise ValueError(f"No agent_end event found in Pi trace: {trace_path}")
    if not messages:
        raise ValueError(f"No completed agent_end run in Pi trace: {trace_path}")
    return messages


def _reject_failed_response(messages: list[dict], trace_path: Path) -> None:
    """Raise when the trace has no final assistant response, or a failed one."""
    final = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if final is None:
        raise ValueError(f"No assistant response in Pi trace: {trace_path}")
    stop_reason = final.get("stopReason")
    if stop_reason in ("error", "aborted"):
        raise ValueError(
            f"Pi trace ended with assistant stopReason={stop_reason!r}, "
            f"skipping: {trace_path}"
        )


def _to_list(values: Any) -> list[int]:
    """Normalize a dataset cell (list, tensor, or array) to a Python list."""
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _tool_arguments_as_mappings(conversation: list[dict]) -> bool:
    """Convert JSON argument strings to mappings for tokenizer chat templates."""
    for turn in conversation:
        for call in turn.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    return False
    return True


def _single_trace_row(  # noqa: PLR0911
    conversation: list[dict],
    tools: list[dict] | None,
    *,
    processor: ProcessorLike,
    render_endpoint: str,
    seq_length: int,
    primary_id: str,
) -> dict | None:
    """Render one full trace and truncate it to the training window."""
    normalized = _normalize_conversation(conversation)
    if not normalized:
        return None
    render_limit = max(seq_length, 32768)
    try:
        boundaries = _render_boundary_rows(
            normalized,
            render_endpoint,
            render_limit,
            tools=tools,
        )
    except (TypeError, ValueError):
        return None
    if not boundaries:
        return None

    tokenizer = getattr(processor, "tokenizer", processor)
    render = getattr(tokenizer, "apply_chat_template", None)
    if render is None:
        return None
    prepared = deepcopy(conversation)
    if not _tool_arguments_as_mappings(prepared):
        return None
    try:
        encoded = render(
            prepared,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
    except (TypeError, ValueError):
        return None
    input_ids = _to_list(encoded.get("input_ids", []))
    assistant_masks = _to_list(encoded.get("assistant_masks", []))
    if not input_ids or len(input_ids) != len(assistant_masks):
        return None

    loss_mask = [1 if value else 0 for value in assistant_masks]
    if not any(loss_mask):
        loss_mask = [0] * len(input_ids)
        for row in boundaries:
            prefix = _to_list(row["input_ids"])
            if input_ids[: len(prefix)] != prefix:
                return None
            for index, value in enumerate(row["loss_mask"]):
                if value:
                    loss_mask[index] = 1
    if not any(loss_mask):
        return None

    input_ids = input_ids[:seq_length]
    loss_mask = loss_mask[:seq_length]
    if not any(loss_mask):
        return None
    decode = getattr(processor, "decode", None)
    text = decode(input_ids) if callable(decode) else tokenizer.decode(input_ids)
    return {
        "id": f"{primary_id}_trace",
        "primary_id": primary_id,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "text": text,
        "metadata": {
            "instance_id": primary_id,
            "source": "pi",
            "row_mode": "per_trace",
            "truncated": len(_to_list(encoded.get("input_ids", []))) > seq_length,
        },
    }


def rows_from_pi_trace(
    trace_path: str | Path,
    *,
    processor: ProcessorLike,
    render_endpoint: str,
    seq_length: int,
    primary_id: str,
    row_mode: str = "per_turn",
) -> list[dict]:
    """Convert one Pi JSONL trace into speculator-format training rows."""
    if row_mode not in ("per_turn", "per_trace"):
        raise ValueError(f"unsupported row_mode: {row_mode}")
    path = Path(trace_path)
    messages = _completed_run_messages(path)
    _reject_failed_response(messages, path)
    conversation, tools = pi_messages_to_conversation(messages)

    if row_mode == "per_trace":
        row = _single_trace_row(
            conversation,
            tools,
            processor=processor,
            render_endpoint=render_endpoint,
            seq_length=seq_length,
            primary_id=primary_id,
        )
        if row is not None:
            return [row]

    sample: dict[str, Any] = {"conversations": conversation}
    if tools is not None:
        # Keep tools as an opaque JSON string: Arrow unions heterogeneous
        # parameter structs when each tool is its own column value, injecting
        # null-valued keys that alter the rendered prompt.
        sample["tools"] = json.dumps(tools)
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
                "text": processor.decode(input_ids),
                "metadata": {
                    "instance_id": primary_id,
                    "source": "pi",
                    **(
                        {"row_mode": "per_turn_fallback"}
                        if row_mode == "per_trace"
                        else {}
                    ),
                },
            }
        )
    return rows
