"""
Vendored DeepSeek-V4 chat template encoding.

Source: deepseek-ai/DeepSeek-V4-Flash, encoding/encoding_dsv4.py
Vendored commit: fd53f944496234770ba80e15004f9b6d269a71f5

Trimmed to encoding-only; parsing/decoding functions removed.
"""

import copy
import json
import re
from typing import Any

__all__ = [
    "DSV4_ASSISTANT_PATTERN",
    "dsv4_format_conversation",
]

# ============================================================
# Special Tokens
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"  # noqa: S105
eos_token: str = "<｜end▁of▁sentence｜>"  # noqa: S105
thinking_start_token: str = "<think>"  # noqa: S105
thinking_end_token: str = "</think>"  # noqa: S105
dsml_token: str = "｜DSML｜"  # noqa: S105

USER_SP_TOKEN = "<｜User｜>"  # noqa: S105
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"  # noqa: S105
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"  # noqa: S105

DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())

# ============================================================
# Templates
# ============================================================

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning_content}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following "
    "schema to reply:\n{schema}"
)
tool_call_template: str = (
    '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
tool_calls_block_name: str = "tool_calls"

tool_output_template: str = "<tool_result>{content}</tool_result>"

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively "
    "decompose the problem to resolve the root cause, rigorously "
    "stress-testing your logic against all potential paths, edge cases, "
    "and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting "
    "every intermediate step, considered alternative, and rejected "
    "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. \
You can invoke tools by writing a "<{dsml_token}tool_calls>" block like \
the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" \
string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. \
For all other types (numbers, booleans, arrays, objects), pass the value \
in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you \
MUST output your complete reasoning inside \
{thinking_start_token}...{thinking_end_token} BEFORE any tool calls or \
final response.

Otherwise, output directly after {thinking_end_token} with tool calls \
or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter \
schemas to invoke tool calls.
"""

# ============================================================
# Utility Functions
# ============================================================


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: dict[str, str]) -> str:
    p_dsml_template = (
        '<{dsml_token}parameter name="{key}" string="{is_str}">'
        "{value}</{dsml_token}parameter>"
    )
    p_dsml_strs: list[str] = []

    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception:
        arguments = {"arguments": tool_call["arguments"]}

    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )
        p_dsml_strs.append(p_dsml_str)

    return "\n".join(p_dsml_strs)


def render_tools(tools: list[dict[str, Any]]) -> str:
    tools_json = [to_json(t) for t in tools]
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: list[dict[str, Any]]) -> int:
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


# ============================================================
# Message Rendering
# ============================================================


def render_message(  # noqa: PLR0912, PLR0915
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    assert 0 <= index < len(messages)
    assert thinking_mode in ["chat", "thinking"], (
        f"Invalid thinking_mode `{thinking_mode}`"
    )

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    assert reasoning_effort in ["max", None, "high"], (
        f"Invalid reasoning effort: {reasoning_effort}"
    )
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
        prompt += REASONING_EFFORT_MAX

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN
        content_developer += content

        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        prompt += USER_SP_TOKEN

        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(
            content=content
        )

    elif role == "tool":
        raise NotImplementedError(
            "deepseek_v4 merges tool messages into user; "
            "please preprocess with merge_tool_messages()"
        )

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls:
            tc_list = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    name=tc.get("name"),
                    arguments=encode_arguments_to_dsml(tc),
                )
                for tc in tool_calls
            ]
            tc_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        rc = reasoning_content or ""

        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (
                    thinking_template.format(reasoning_content=rc) + thinking_end_token
                )
            else:
                thinking_part = ""

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in [
        "assistant",
        "latest_reminder",
    ]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        assert task in VALID_TASKS, (
            f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        )
        task_sp_token = DS_TASK_SP_TOKENS[task]

        if task != "action":
            prompt += task_sp_token
        else:
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                thinking_end_token
                if thinking_mode != "thinking"
                else thinking_start_token
            )
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"]:
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":  # noqa: SIM114
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt


# ============================================================
# Preprocessing
# ============================================================


def merge_tool_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)  # noqa: PLW2901
        role = msg.get("role")

        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append(
                    {
                        "role": "user",
                        "content_blocks": [tool_block],
                    }
                )
        elif role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_msg = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [text_block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    last_tool_call_order: dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0),
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# Main Encoding Function
# ============================================================


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    context = context if context else []

    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def _drop_thinking_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {
        "user",
        "system",
        "tool",
        "latest_reminder",
        "direct_search_results",
    }

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)  # noqa: PLW2901
            msg.pop("reasoning_content", None)
            result.append(msg)

    return result


# ============================================================
# Integration API
# ============================================================

DSV4_ASSISTANT_PATTERN = re.escape(thinking_end_token) + r"(.*?)" + re.escape(eos_token)


def dsv4_format_conversation(conversation: list[dict[str, Any]]) -> str:
    """Format a normalized conversation using the DeepSeek-V4 encoding.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys.

    Returns:
        Formatted conversation string with DSv4 special tokens.
    """
    return encode_messages(messages=conversation, thinking_mode="chat")
