#!/usr/bin/env python3
"""Convert native Verifiers traces into Speculators trajectory JSONL."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import tomllib
import verifiers.v1 as vf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Verifiers eval traces into exact-token, "
            "Speculators-compatible trajectories."
        )
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="Native Verifiers traces.jsonl files",
    )
    parser.add_argument(
        "--outfile",
        default="output/agentic_regen/trajectories.jsonl",
        help="Speculators-compatible trajectory JSONL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model metadata (otherwise read from the run config)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override endpoint metadata (otherwise read from the run config)",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Override environment metadata (otherwise read from the run config)",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Export partial errored traces in addition to completed traces",
    )
    return parser.parse_args()


def _content_to_json(content: Any) -> Any:
    if isinstance(content, list):
        return [part.model_dump(mode="json") for part in content]
    return content


def message_to_openai(message: vf.Message) -> dict[str, Any]:
    """Convert a Verifiers message without losing tool-call linkage."""
    if isinstance(message, vf.AssistantMessage):
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.reasoning_content:
            result["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return result
    if isinstance(message, vf.ToolMessage):
        result = {
            "role": "tool",
            "content": _content_to_json(message.content),
            "tool_call_id": message.tool_call_id,
        }
        if message.name:
            result["name"] = message.name
        return result
    return {
        "role": message.role,
        "content": _content_to_json(message.content),
    }


def tool_to_openai(tool: vf.Tool) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def trace_to_dataset_records(
    trace: vf.Trace, *, model: str, endpoint: str, environment_id: str
) -> list[dict[str, Any]]:
    """Export one record for every root-to-leaf branch in a Verifiers trace."""
    # Stable Verifiers stores exact rendered tokens and messages, but does not
    # duplicate tool schemas on disk. Preserve them if a future release adds them.
    tools = [
        tool_to_openai(tool)
        for tool in getattr(trace, "tools", None) or []
    ]
    usage = trace.usage.model_dump(mode="json") if trace.usage else None
    records = []
    for branch in trace.branches:
        conversations = [message_to_openai(message) for message in branch.messages]
        if not conversations:
            continue
        token_ids = branch.token_ids or []
        sampled_mask = branch.sampled_mask or []
        has_exact_tokens = bool(token_ids) and len(token_ids) == len(sampled_mask)
        records.append(
            {
                "id": f"{trace.id}:branch-{branch.index}",
                "conversations": conversations,
                "tools": tools,
                "input_ids": token_ids if has_exact_tokens else None,
                "loss_mask": sampled_mask if has_exact_tokens else None,
                "metadata": {
                    "source": "verifiers",
                    "environment": environment_id,
                    "trace_id": trace.id,
                    "branch_index": branch.index,
                    "model": model,
                    "endpoint": endpoint,
                    "on_policy": True,
                    "reward": trace.reward,
                    "rewards": trace.rewards,
                    "metrics": trace.metrics,
                    "num_turns": trace.num_turns,
                    "stop_condition": trace.stop_condition,
                    "usage": usage,
                    "verifiers_token_ids_available": has_exact_tokens,
                    "token_count": len(token_ids),
                    "sampled_token_count": sum(sampled_mask),
                },
            }
        )
    return records


def _run_metadata(path: Path) -> tuple[str, str, str]:
    """Read taskset, model, and endpoint from the CLI's sibling config.toml."""
    config_path = path.parent / "config.toml"
    if not config_path.exists():
        return "", "", ""
    try:
        with config_path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Invalid Verifiers run config: {config_path}") from error

    taskset = config.get("taskset", {})
    client = config.get("client", {})
    return (
        str(taskset.get("id", "")),
        str(config.get("model", "")),
        str(client.get("base_url", "")),
    )


def iter_native_traces(
    paths: list[str],
) -> Iterator[tuple[vf.Trace, str, str, str]]:
    """Read Verifiers 0.2.0 native trace records and their run metadata."""
    for path in paths:
        trace_path = Path(path)
        environment, model, endpoint = _run_metadata(trace_path)
        with trace_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    yield (
                        vf.WireTrace.model_validate(record),
                        environment,
                        model,
                        endpoint,
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid Verifiers record at {path}:{line_number}"
                    ) from error


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = []
    trace_count = 0
    skipped_count = 0
    for trace, native_environment, native_model, native_endpoint in (
        iter_native_traces(args.traces)
    ):
        trace_count += 1
        if not args.include_errors and (
            not trace.is_completed or trace.has_error
        ):
            skipped_count += 1
            continue
        rows.extend(
            trace_to_dataset_records(
                trace,
                model=args.model or native_model,
                endpoint=args.endpoint or native_endpoint,
                environment_id=args.environment or native_environment,
            )
        )

    if not rows:
        raise RuntimeError("No trajectory branches were exported")
    exact_rows = sum(
        row["metadata"]["verifiers_token_ids_available"] for row in rows
    )
    if exact_rows != len(rows):
        raise RuntimeError(
            "Some branches lack exact token IDs; rerun eval with client.type=train"
        )

    write_jsonl(args.outfile, rows)
    print(f"Read traces: {trace_count} (skipped: {skipped_count})")
    print(f"Exported branches: {len(rows)}")
    print(f"Exact-token branches: {exact_rows}/{len(rows)}")
    print(f"Trajectories: {args.outfile}")


if __name__ == "__main__":
    main()
