"""Collect time-bounded concurrency curves with GuideLLM."""

from __future__ import annotations

import argparse  # noqa: TC003
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx


def refuse_prefix_caching(target: str) -> None:
    """Require uncached prefill because the sweep reuses its prompts."""
    response = httpx.get(target + "/metrics", timeout=10, trust_env=False)
    response.raise_for_status()
    flag = re.search(
        r'^vllm:cache_config_info\{[^\n]*enable_prefix_caching="([^"]+)"',
        response.text,
        re.MULTILINE,
    )
    if flag is None or flag[1].lower() != "false":
        raise ValueError("Serve vLLM with --no-enable-prefix-caching for speedup runs")


def summarize_benchmark(report: dict) -> dict:
    """Use GuideLLM's window throughput and per-request, token-normalized TPOT."""
    (benchmark,) = report["benchmarks"]
    constraints = benchmark["scheduler_state"]["end_processing_constraints"]
    if "max_duration" not in constraints or "requests_exhausted" in constraints:
        raise ValueError(
            "Timed point ended early before completing its duration window"
        )
    requests = benchmark["requests"]
    if requests["errored"]:
        raise ValueError("Cannot use a benchmark with failed requests")
    timing = benchmark["scheduler_metrics"]
    start, end = timing["measure_start_time"], timing["measure_end_time"]
    measured = [
        r for r in requests["successful"] if start <= r["request_end_time"] <= end
    ]
    tpots = [
        r["inter_token_latency_ms"] / 1000
        for r in measured
        if r["output_tokens"] > 1 and r["inter_token_latency_ms"] is not None
    ]
    if not tpots or end <= start:
        raise ValueError("No usable requests completed in the measurement window")
    x = 1 / statistics.fmean(tpots)
    y = benchmark["metrics"]["output_tokens_per_second"]["total"]["mean"]
    if not all(math.isfinite(v) and v > 0 for v in (x, y)):
        raise ValueError("Invalid throughput or interactivity")
    return {
        "streams": benchmark["config"]["strategy"]["streams"],
        "x": x,
        "y": y,
        "completed": len(measured),
        "duration": end - start,
        "incomplete": len(requests["incomplete"]),
    }


def _run_point(command: list[str], result_path: Path, target: str) -> bool:
    try:
        refuse_prefix_caching(target)
        returncode = subprocess.run(command, check=False).returncode  # noqa: S603
        if returncode:
            print(
                f"{result_path.name}: GuideLLM exited with {returncode}",
                file=sys.stderr,
            )
            return False
        refuse_prefix_caching(target)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"{result_path.name}: {exc}", file=sys.stderr)
        return False
    return True


def read_report(path: Path) -> dict | None:
    try:
        report = json.loads(path.read_text())
        if "benchmarks" in report:
            summarize_benchmark(report)
        elif report["completed"] != report["num_prompts"]:
            raise ValueError("Cannot use a benchmark with failed requests")
        return report
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"Warning: skipping {path.name}: {exc}", file=sys.stderr)
        return None


def _write_results(args: argparse.Namespace, failures: int) -> int:
    groups = defaultdict(list)
    for path in sorted(args.output_dir.glob("conc*_r*.json")):
        report = read_report(path)
        if report is None:
            continue
        report["label"] = args.label
        path.write_text(json.dumps(report))
        point = summarize_benchmark(report)
        groups[point["streams"]].append(point)
    rows = [
        {
            "concurrency": streams,
            "repeats": len(points),
            "interactivity_tok_s_per_user": statistics.fmean(p["x"] for p in points),
            "output_throughput_tok_s": statistics.fmean(p["y"] for p in points),
        }
        for streams, points in sorted(groups.items())
    ]
    if not rows:
        return 1
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return int(bool(failures))


def run(args: argparse.Namespace, requests: list[dict]) -> int:
    with TemporaryDirectory(prefix="speculators-speedup-") as temporary:
        dataset_path = Path(temporary) / "prompts.jsonl"
        # This is an input supply, not a quota: max_duration ends each point.
        copies = max(1, math.ceil(5000 / len(requests)))
        with dataset_path.open("w") as handle:
            for _ in range(copies):
                for request in requests:
                    handle.write(
                        json.dumps({"messages": request["body"]["messages"]}) + "\n"
                    )

        body = dict(requests[0]["body"])
        for key in ("model", "messages", "stream", "stream_options"):
            body.pop(key, None)
        backend = {
            "kind": "openai_http",
            "target": args.target,
            "model": args.model,
            "request_format": "/v1/chat/completions",
            "max_tokens": body.pop(
                "max_tokens", body.pop("max_completion_tokens", None)
            ),
            "extras": {"body": body},
        }
        duration = args.warmup_seconds + args.max_seconds
        failures = 0
        for streams in args.streams:
            for repeat in range(1, args.repeats + 1):
                name = f"conc{streams}_r{repeat}"
                result_path = args.output_dir / f"{name}.json"
                if result_path.exists() and not args.overwrite:
                    continue
                result_path.unlink(missing_ok=True)
                command = [
                    args.guidellm_bin,
                    "run",
                    "--backend",
                    json.dumps(backend),
                    "--tokenizer",
                    json.dumps(
                        {"kind": "hf_auto", "model": args.tokenizer or args.model}
                    ),
                    "--data",
                    json.dumps(
                        {
                            "kind": "json_file",
                            "path": str(dataset_path),
                            "load_kwargs": {"split": "train"},
                        }
                    ),
                    "--data-column-mapper",
                    "kind=generative_column_mapper,column_mappings.raw_messages_column=messages",
                    "--profile",
                    f"kind=concurrent,streams={streams},warmup={args.warmup_seconds:g}",
                    "--constraint",
                    f"kind=max_duration,seconds={duration:g}",
                    "--metrics",
                    "kind=generative,sample_size=0,prefer_response_metrics=false",
                    "--label",
                    f"name={args.label}",
                    "--output",
                    json.dumps({"kind": "json", "path": str(result_path)}),
                    "--disable-console-interactive",
                ]
                print(f"Collecting {name}", flush=True)
                if not _run_point(command, result_path, args.target):
                    failures += 1
                    if not args.keep_going:
                        return 1
        return _write_results(args, failures)
