"""Tests for Pi trace conversion into speculator-format rows."""

import json

import pytest
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


def _capture_conversations(tmp_path, monkeypatch, events):
    """Run a synthetic trace through the builder and return its conversation."""
    trace_path = tmp_path / "pi.jsonl"
    trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    captured = {}

    def fake_build(dataset, processor, max_length, num_proc, **kwargs):
        captured["dataset"] = dataset
        return HFDataset.from_list([])

    monkeypatch.setattr(agentic_traces, "build_speculator_training_dataset", fake_build)
    rows_from_pi_trace(
        trace_path,
        processor=_FakeProcessor(),
        render_endpoint="http://localhost:8000",
        seq_length=64,
        primary_id="inst.abc",
    )
    return captured["dataset"][0]["conversations"]


def test_rows_from_pi_trace_accumulates_runs(tmp_path, monkeypatch):
    first_run = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "step one"}],
            "stopReason": "toolUse",
        },
        {"role": "toolResult", "toolCallId": "call_1", "content": "ok"},
    ]
    # Compaction re-emits the system prompt and starts a new run.
    second_run = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "step two"}],
            "stopReason": "stop",
        },
    ]
    events = [
        {"type": "agent_start"},
        {"type": "agent_end", "messages": first_run, "willRetry": False},
        {"type": "agent_start"},
        {"type": "agent_end", "messages": second_run, "willRetry": False},
    ]

    conversation = _capture_conversations(tmp_path, monkeypatch, events)

    assert [(turn["role"], turn["content"]) for turn in conversation] == [
        ("system", "sys"),
        ("user", "task"),
        ("assistant", "step one"),
        ("tool", "ok"),
        ("assistant", "step two"),
    ]


def test_rows_from_pi_trace_keeps_prompt_from_failed_retry_attempt(
    tmp_path, monkeypatch
):
    # A retry continues the previous run, so its agent_end only carries the new
    # assistant message. The prompt from the failed attempt must survive.
    events = [
        {
            "type": "agent_end",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "stopReason": "error"},
            ],
            "willRetry": True,
        },
        {
            "type": "agent_end",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                    "stopReason": "stop",
                }
            ],
            "willRetry": False,
        },
    ]

    conversation = _capture_conversations(tmp_path, monkeypatch, events)

    assert [turn["content"] for turn in conversation] == ["sys", "task", "answer"]


def test_rows_from_pi_trace_rejects_failed_final_response(tmp_path, monkeypatch):
    events = [
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "stopReason": "error"},
            ],
            "willRetry": False,
        }
    ]

    with pytest.raises(ValueError, match="stopReason='error'"):
        _capture_conversations(tmp_path, monkeypatch, events)


def test_rows_from_pi_trace_rejects_aborted_final_response(tmp_path, monkeypatch):
    events = [
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "stopReason": "aborted"},
            ],
            "willRetry": False,
        }
    ]

    with pytest.raises(ValueError, match="stopReason='aborted'"):
        _capture_conversations(tmp_path, monkeypatch, events)


def test_rows_from_pi_trace_requires_a_completed_run(tmp_path, monkeypatch):
    events = [
        {
            "type": "agent_end",
            "messages": [{"role": "assistant", "content": "", "stopReason": "error"}],
            "willRetry": True,
        }
    ]

    with pytest.raises(ValueError, match="No completed agent_end run"):
        _capture_conversations(tmp_path, monkeypatch, events)


def test_rows_from_pi_trace_rejects_prompt_without_assistant(tmp_path, monkeypatch):
    # The retry never finished, so the trace holds only the failed attempt's prompt.
    events = [
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "stopReason": "error"},
            ],
            "willRetry": True,
        }
    ]

    with pytest.raises(ValueError, match="No assistant response"):
        _capture_conversations(tmp_path, monkeypatch, events)


def test_rows_from_pi_trace_keeps_other_stop_reasons(tmp_path, monkeypatch):
    events = [
        {
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "partial"}],
                    "stopReason": "length",
                },
            ],
            "willRetry": False,
        }
    ]

    conversation = _capture_conversations(tmp_path, monkeypatch, events)

    assert conversation[-1]["content"] == "partial"


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
    # Both agent_end events describe the same conversation, so dedup leaves it.
    assert captured["dataset"][0]["conversations"][-1]["role"] == "tool"
    assert isinstance(captured["dataset"][0]["tools"], str)
    parsed_tools = json.loads(captured["dataset"][0]["tools"])
    assert parsed_tools == _pi_messages()[0]["toolsAdded"]

    assert [row["id"] for row in rows] == ["inst.abc_gen0", "inst.abc_gen1"]
    assert [row["primary_id"] for row in rows] == ["inst.abc", "inst.abc"]
    assert rows[0]["input_ids"] == [1, 2, 3]
    assert rows[0]["loss_mask"] == [0, 1, 1]
    assert rows[1]["loss_mask"] == [0, 1]
    assert rows[0]["text"] == "<3 tokens>"
    assert rows[0]["metadata"] == {"instance_id": "inst.abc", "source": "pi"}


def test_rows_from_pi_trace_preserves_heterogeneous_tools(tmp_path, monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                },
            },
        },
    ]
    messages = [
        {"role": "system", "content": "sys", "toolsAdded": tools},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    trace_path = tmp_path / "pi.jsonl"
    trace_path.write_text(
        json.dumps({"type": "agent_end", "messages": messages}) + "\n"
    )

    captured = {}

    def fake_build(dataset, processor, max_length, num_proc, **kwargs):
        captured["dataset"] = dataset
        return HFDataset.from_list([])

    monkeypatch.setattr(agentic_traces, "build_speculator_training_dataset", fake_build)

    rows_from_pi_trace(
        trace_path,
        processor=_FakeProcessor(),
        render_endpoint="http://localhost:8000",
        seq_length=64,
        primary_id="inst.abc",
    )

    cell = captured["dataset"][0]["tools"]
    assert isinstance(cell, str)
    parsed = json.loads(cell)
    assert parsed == tools
    for tool, original in zip(parsed, tools, strict=True):
        assert (
            tool["function"]["parameters"]["properties"]
            == (original["function"]["parameters"]["properties"])
        )
