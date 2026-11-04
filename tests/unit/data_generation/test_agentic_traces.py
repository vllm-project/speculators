"""Tests for Pi trace conversion into speculator-format rows."""

import json

from datasets import Dataset as HFDataset

from speculators.data_generation import agentic_traces
from speculators.data_generation.agentic_traces import (
    pi_messages_to_conversation,
    rows_from_pi_trace,
)


def _pi_messages():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    return [
        {
            "role": "system",
            "content": "",
            "sections": {
                "preamble": "You are pi.",
                "tools": "<tools>bash</tools>",
                "rules": "<rules>be brief</rules>",
            },
            "toolsAdded": tools,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "!"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "think"},
                {"type": "text", "text": "running"},
                {
                    "type": "toolCall",
                    "id": "call_1",
                    "name": "bash",
                    "arguments": {"command": "ls"},
                },
                {
                    "type": "toolCall",
                    "id": "call_2",
                    "name": "bash",
                    "arguments": {"command": "pwd"},
                },
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "call_1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "a.py"}],
        },
        {
            "role": "toolResult",
            "toolCallId": "call_2",
            "toolName": "bash",
            "content": [{"type": "text", "text": "/testbed"}],
        },
    ]


def test_pi_messages_to_conversation_flattens_everything():
    conversation, tools = pi_messages_to_conversation(_pi_messages())

    assert [turn["role"] for turn in conversation] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert conversation[0]["content"] == (
        "You are pi.\n\n<tools>bash</tools>\n\n<rules>be brief</rules>"
    )
    assert conversation[1]["content"] == "hello!"
    assert tools == _pi_messages()[0]["toolsAdded"]

    assistant = conversation[2]
    assert assistant["content"] == "running"
    assert assistant["thinking"] == "think"
    assert assistant["reasoning_content"] == "think"
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
    assert all(call["type"] == "function" for call in assistant["tool_calls"])
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }
    assert assistant["tool_calls"][1]["function"]["name"] == "bash"

    assert conversation[3] == {
        "role": "tool",
        "content": "a.py",
        "tool_call_id": "call_1",
    }
    assert conversation[4]["content"] == "/testbed"


def test_pi_messages_to_conversation_wraps_pi_tool_schema():
    messages = [
        {
            "role": "system",
            "content": "sys",
            "toolsAdded": [
                {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
    ]
    _, tools = pi_messages_to_conversation(messages)
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_pi_messages_to_conversation_without_tools():
    conversation, tools = pi_messages_to_conversation(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    )
    assert tools is None
    assert conversation == [{"role": "user", "content": "hi"}]


class _FakeProcessor:
    def decode(self, input_ids):
        return f"<{len(input_ids)} tokens>"


def test_rows_from_pi_trace_uses_builder(tmp_path, monkeypatch):
    trace_path = tmp_path / "pi.jsonl"
    events = [
        {"type": "agent_start"},
        {"type": "agent_end", "messages": _pi_messages()[:2]},
        {"type": "agent_end", "messages": _pi_messages()},
    ]
    trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    captured = {}

    def fake_build(dataset, processor, max_length, num_proc, **kwargs):
        captured["dataset"] = dataset
        captured["max_length"] = max_length
        captured["num_proc"] = num_proc
        captured.update(kwargs)
        return HFDataset.from_list(
            [
                {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
                {"input_ids": [1, 4], "loss_mask": [0, 1]},
            ]
        )

    monkeypatch.setattr(agentic_traces, "build_speculator_training_dataset", fake_build)

    rows = rows_from_pi_trace(
        trace_path,
        processor=_FakeProcessor(),
        render_endpoint="http://localhost:8000",
        seq_length=64,
        primary_id="inst.abc",
    )

    assert captured["max_length"] == 64
    assert captured["num_proc"] == 1
    assert captured["render_endpoint"] == "http://localhost:8000"
    assert captured["minimum_valid_tokens"] == 1
    # The last agent_end event wins, so the full conversation was converted.
    assert captured["dataset"][0]["conversations"][-1]["role"] == "tool"
    assert captured["dataset"][0]["tools"] == _pi_messages()[0]["toolsAdded"]

    assert [row["id"] for row in rows] == ["inst.abc_gen0", "inst.abc_gen1"]
    assert [row["primary_id"] for row in rows] == ["inst.abc", "inst.abc"]
    assert rows[0]["input_ids"] == [1, 2, 3]
    assert rows[0]["loss_mask"] == [0, 1, 1]
    assert rows[1]["loss_mask"] == [0, 1]
    assert rows[0]["text"] == "<3 tokens>"
    assert rows[0]["metadata"] == {"instance_id": "inst.abc", "source": "pi"}
