"""Unit tests for native Verifiers trace conversion and exact-token replay."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("verifiers.v1")

import verifiers.v1 as vf

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "agentic_regeneration"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "agentic_regeneration_converter", SCRIPT_DIR / "convert_traces.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_export_preserves_tool_call_linkage():
    script = _load_script()
    trace = vf.Trace(
        id="trace-1",
        task=vf.TraceTask(
            type="ArithmeticTask",
            data=vf.TaskData(idx=0, prompt="Use the tool"),
        ),
        nodes=[
            vf.MessageNode(
                message=vf.UserMessage(content="Use the tool"),
                token_ids=[1, 2],
                mask=[False, False],
            ),
            vf.MessageNode(
                parent=0,
                sampled=True,
                token_ids=[3, 4],
                mask=[True, True],
                message=vf.AssistantMessage(
                    content=None,
                    tool_calls=[
                        vf.ToolCall(
                            id="call-1",
                            name="multiply_and_add",
                            arguments='{"a":7,"b":8,"c":9}',
                        )
                    ],
                ),
            ),
            vf.MessageNode(
                parent=1,
                token_ids=[5],
                mask=[False],
                message=vf.ToolMessage(
                    tool_call_id="call-1",
                    content="65",
                ),
            ),
            vf.MessageNode(
                parent=2,
                sampled=True,
                token_ids=[6],
                mask=[True],
                message=vf.AssistantMessage(content="65"),
            ),
        ],
        is_completed=True,
        rewards={"correct_after_tool": 1.0},
    )

    (row,) = script.trace_to_dataset_records(
        trace,
        model="Qwen/Qwen3-0.6B",
        endpoint="http://localhost:8000/v1",
        environment_id="agentic_regen_env",
    )

    assert [message["role"] for message in row["conversations"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["conversations"][1]["tool_calls"][0]["id"] == "call-1"
    assert row["conversations"][2]["tool_call_id"] == "call-1"
    assert row["tools"] == []
    assert row["metadata"]["on_policy"] is True
    assert row["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert row["loss_mask"] == [False, False, True, True, False, True]
    assert row["metadata"]["verifiers_token_ids_available"] is True


def test_native_trace_reader_uses_sibling_run_config(tmp_path):
    script = _load_script()
    trace = vf.Trace(
        id="trace-1",
        task=vf.TraceTask(
            type="Task", data=vf.TaskData(idx=0, prompt="hello")
        ),
        nodes=[
            vf.MessageNode(
                message=vf.UserMessage(content="hello"),
                token_ids=[1],
                mask=[False],
            )
        ],
        is_completed=True,
    )
    path = tmp_path / "traces.jsonl"
    path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        'model = "Qwen/Qwen3-8B"\n'
        "[taskset]\n"
        'id = "upstream-v1"\n'
        "[client]\n"
        'base_url = "http://localhost:8000/v1"\n',
        encoding="utf-8",
    )

    ((loaded_trace, environment_id, model, endpoint),) = (
        script.iter_native_traces([str(path)])
    )

    assert environment_id == "upstream-v1"
    assert model == "Qwen/Qwen3-8B"
    assert endpoint == "http://localhost:8000/v1"
    assert loaded_trace.branches[0].token_ids == [1]
