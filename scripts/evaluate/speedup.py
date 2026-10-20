"""Collect time-bounded concurrency curves with GuideLLM."""

from __future__ import annotations

import argparse  # noqa: TC003
import csv
import fcntl
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread

import httpx

MIN_USABLE_COMPLETIONS = 10
MAX_REPEAT_CV = 0.1


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


class ServerMetrics:
    """Observe queue gauges during a point and difference speculative counters."""

    NAMES = {
        "waiting": "vllm:num_requests_waiting",
        "steps": "vllm:spec_decode_num_drafts_total",
        "drafted": "vllm:spec_decode_num_draft_tokens_total",
        "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    }

    def __init__(self, target: str, model: str):
        self.target, self.model = target, model
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.stopped = Event()
        self.thread = Thread(target=self._poll, daemon=True)

    def _sample(self) -> None:
        try:
            response = httpx.get(self.target + "/metrics", timeout=2, trust_env=False)
            response.raise_for_status()
            values: dict[str, float] = {}
            for line in response.text.splitlines():
                match = re.match(r"^(vllm:\w+)(?:\{([^}]*)\})?\s+(\S+)", line)
                if not match:
                    continue
                labels = dict(re.findall(r'(\w+)="([^"\\]*)"', match[2] or ""))
                if labels.get("model_name", self.model) != self.model:
                    continue
                for key, name in self.NAMES.items():
                    if match[1] == name:
                        value = float(match[3])
                        if not math.isfinite(value):
                            raise ValueError(f"Non-finite server metric: {name}")
                        values[key] = values.get(key, 0) + value
            self.samples.append({"time": time.time(), **values})
        except (httpx.HTTPError, ValueError) as exc:
            self.errors.append(str(exc))

    def _poll(self) -> None:
        while not self.stopped.wait(2):
            self._sample()

    def start(self) -> None:
        self._sample()
        self.thread.start()

    def finish(self) -> dict:
        self.stopped.set()
        self.thread.join()
        self._sample()
        result: dict = {
            "scope": "whole_point_including_warmup_and_client_setup",
            "samples": self.samples,
            "scrape_errors": self.errors,
        }
        waiting = [s["waiting"] for s in self.samples if "waiting" in s]
        result["waiting_max"] = max(waiting) if waiting else None
        result["waiting_mean"] = statistics.fmean(waiting) if waiting else None
        if self.samples:
            before, after = self.samples[0], self.samples[-1]
            for key in ("steps", "drafted", "accepted"):
                if key in before and key in after and after[key] >= before[key]:
                    result[key] = after[key] - before[key]
        steps, accepted = result.get("steps"), result.get("accepted")
        result["acceptance_length"] = (
            1 + accepted / steps if steps and accepted is not None else None
        )
        return result


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
    ttfts = [
        r["time_to_first_token_ms"]
        for r in measured
        if r.get("time_to_first_token_ms") is not None
    ]
    return {
        "streams": benchmark["config"]["strategy"]["streams"],
        "x": x,
        "y": y,
        "completed": len(measured),
        "usable_completions": len(tpots),
        "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else None,
        "duration": end - start,
        "incomplete": len(requests["incomplete"]),
    }


def read_archive(path: Path) -> tuple[dict, dict[str, dict], int]:
    """Read the last attempt per point; tolerate only an interrupted final write."""
    data = path.read_bytes()
    records, offset = [], 0
    for line in data.splitlines(keepends=True):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if offset + len(line) != len(data) or line.endswith(b"\n"):
                raise ValueError(
                    f"Malformed benchmark archive at byte {offset}"
                ) from None
            print("Warning: ignoring interrupted final archive write", file=sys.stderr)
            break
        offset += len(line)
    if not records:
        raise ValueError("Benchmark archive has no input header")
    return records[0], {r["point"]: r for r in records[1:]}, offset


def write_summary(
    path: Path, records: dict[str, dict], streams: list[int], repeats: int
) -> None:
    """Retain failed counts and diagnostics; average axes only over valid repeats."""
    rows = []
    for n in streams:
        attempts = [
            records[f"conc{n}_r{i}"]
            for i in range(1, repeats + 1)
            if f"conc{n}_r{i}" in records
        ]
        if not attempts:
            continue
        good = [r for r in attempts if r.get("status") == "ok"]
        points = [r["summary"] for r in good]
        diagnostics = [r.get("diagnostics", {}) for r in good]
        row: dict = {
            "concurrency": n,
            "repeats": len(good),
            "interactivity_tok_s_per_user": statistics.fmean(p["x"] for p in points)
            if points
            else None,
            "output_throughput_tok_s": statistics.fmean(p["y"] for p in points)
            if points
            else None,
            "expected_repeats": repeats,
            "failed_repeats": sum(r.get("status") != "ok" for r in attempts),
            "status": "complete"
            if len(good) == repeats
            else "partial"
            if good
            else "failed",
            "min_usable_completions": min(
                (p["usable_completions"] for p in points), default=None
            ),
        }
        for key in ("x", "y"):
            values = [p[key] for p in points]
            row[f"{key}_repeat_cv"] = (
                statistics.pstdev(values) / statistics.fmean(values)
                if len(values) > 1
                else None
            )
        ttfts = [p["mean_ttft_ms"] for p in points if p["mean_ttft_ms"] is not None]
        row["mean_ttft_ms"] = statistics.fmean(ttfts) if ttfts else None
        waiting = [
            d["waiting_max"] for d in diagnostics if d.get("waiting_max") is not None
        ]
        row["waiting_requests_max"] = max(waiting) if waiting else None
        specs = [
            d
            for d in diagnostics
            if d.get("steps") is not None and d.get("accepted") is not None
        ]
        steps = sum(d["steps"] for d in specs)
        row["acceptance_length"] = (
            1 + sum(d["accepted"] for d in specs) / steps if steps else None
        )
        rows.append(row)
    if rows:
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def _append(archive, record: dict) -> None:
    archive.write(json.dumps(record, allow_nan=False) + "\n")
    archive.flush()
    os.fsync(archive.fileno())


@contextmanager
def _journal(archive_path: Path, header: dict, resume: bool):
    with archive_path.open("a+") as archive:
        try:
            fcntl.flock(archive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This sweep already has an active writer") from None
        archive.seek(0, os.SEEK_END)
        if not resume and archive.tell():
            raise ValueError("Benchmark archive already exists; use --resume")
        records: dict[str, dict] = {}
        if resume:
            saved, records, offset = read_archive(archive_path)
            if saved != header:
                raise ValueError(
                    "Resume configuration or inputs differ; "
                    "use a fresh output directory"
                )
            archive.seek(offset)
            archive.truncate()
            if offset and not archive_path.read_bytes().endswith(b"\n"):
                archive.write("\n")
        if not resume:
            _append(archive, header)
        yield archive, records


def _command(args, backend, dataset_path, result_path, streams) -> list[str]:
    duration = args.warmup_seconds + args.max_seconds
    return [
        args.guidellm_bin,
        "run",
        "--backend",
        json.dumps(backend),
        "--tokenizer",
        json.dumps({"kind": "hf_auto", "model": args.tokenizer or args.model}),
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


def _collect_point(args, name, command, result_path, archive) -> dict:
    record = {"point": name, "command": command, "status": "started"}
    _append(archive, record)
    monitor = ServerMetrics(args.target, args.model)
    monitor.start()
    print(f"Collecting {name}", flush=True)
    try:
        # An orphaned client keeps the lock, preventing a competing resume.
        subprocess.run(command, check=True, pass_fds=(archive.fileno(),))  # noqa: S603
        record["report"] = json.loads(result_path.read_text())
        record["report"]["label"] = args.label
        record["summary"] = summarize_benchmark(record["report"])
        record["status"] = "ok"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        record["diagnostics"] = monitor.finish()
        if "report" not in record and result_path.exists():
            try:
                record["report"] = json.loads(result_path.read_text())
            except ValueError:
                record["unparsed_report"] = result_path.read_text()
        if record["status"] == "started":
            record.update(
                status="interrupted",
                error="Point interrupted; retry only with --resume --retry-failed",
            )
        _append(archive, record)
    return record


def _warn_point(record: dict) -> None:
    name = record["point"]
    notices = []
    if record["status"] != "ok":
        notices.append(record["error"])
    elif record["summary"]["usable_completions"] < MIN_USABLE_COMPLETIONS:
        notices.append(
            f"only {record['summary']['usable_completions']} usable completions; "
            "increase --max-seconds for a more stable x"
        )
    waiting = record["diagnostics"]["waiting_max"]
    if waiting:
        notices.append(
            f"waiting requests observed (sampled max={waiting:g}); "
            "inspect scheduler/KV capacity"
        )
    if record["diagnostics"]["scrape_errors"] or waiting is None:
        notices.append("server diagnostics incomplete; inspect archived scrape errors")
    for notice in notices:
        print(f"Warning: {name}: {notice}", file=sys.stderr, flush=True)


def _warn_repeats(records, streams, repeats) -> None:
    valid = [
        records[f"conc{streams}_r{i}"]["summary"]
        for i in range(1, repeats + 1)
        if records.get(f"conc{streams}_r{i}", {}).get("status") == "ok"
    ]
    for axis in ("x", "y"):
        values = [p[axis] for p in valid]
        if (
            len(values) > 1
            and statistics.pstdev(values) / statistics.fmean(values) > MAX_REPEAT_CV
        ):
            print(
                f"Warning: concurrency {streams}: "
                f"{axis} varies by more than 10% across repeats",
                file=sys.stderr,
            )


def run(args: argparse.Namespace, requests: list[dict]) -> int:
    """Journal attempts, continue failed points, and explicitly resume missing work."""
    archive_path = args.output_dir / "benchmarks.jsonl"
    resume = getattr(args, "resume", False)
    if archive_path.exists() and not resume:
        raise ValueError(
            f"{args.output_dir} already exists; "
            "use --resume or a fresh output directory"
        )
    if resume and not archive_path.exists():
        raise ValueError("--resume requires an existing benchmarks.jsonl")
    if (args.output_dir / "artifacts").exists() or any(
        args.output_dir.glob("conc*_r*.json")
    ):
        raise ValueError("Use a fresh output directory for legacy per-point results")
    inputs = [{"messages": r["body"]["messages"]} for r in requests]
    body = dict(requests[0]["body"])
    for key in ("model", "messages", "stream", "stream_options"):
        body.pop(key, None)
    backend = {
        "kind": "openai_http",
        "target": args.target,
        "model": args.model,
        "request_format": "/v1/chat/completions",
        "max_tokens": body.pop("max_tokens", body.pop("max_completion_tokens", None)),
        "extras": {"body": body},
    }
    config = {
        "backend": backend,
        "tokenizer": args.tokenizer or args.model,
        "label": args.label,
        "streams": args.streams,
        "repeats": args.repeats,
        "warmup_seconds": args.warmup_seconds,
        "max_seconds": args.max_seconds,
    }
    header: dict = {
        "inputs": inputs,
        "copies": max(1, math.ceil(5000 / len(requests))),
        "config": config,
    }
    with _journal(archive_path, header, resume) as (archive, records):
        refuse_prefix_caching(args.target)
        with TemporaryDirectory(prefix="speculators-speedup-") as temporary:
            dataset_path = Path(temporary) / "prompts.jsonl"
            with dataset_path.open("w") as handle:
                for _ in range(header["copies"]):
                    for request in inputs:
                        handle.write(json.dumps(request) + "\n")
            print(
                f"Sweep: {len(args.streams) * args.repeats} points, "
                f"{args.warmup_seconds:g}s warmup + {args.max_seconds:g}s measured "
                "each (plus setup/teardown)",
                flush=True,
            )
            for streams in args.streams:
                for repeat in range(1, args.repeats + 1):
                    name = f"conc{streams}_r{repeat}"
                    previous = records.get(name)
                    if previous and (
                        previous.get("status") == "ok"
                        or not getattr(args, "retry_failed", False)
                    ):
                        print(
                            f"Skipping recorded {name} "
                            f"({previous.get('status', 'interrupted')})",
                            flush=True,
                        )
                        continue
                    result_path = Path(temporary) / f"{name}.json"
                    command = _command(
                        args, backend, dataset_path, result_path, streams
                    )
                    record = _collect_point(args, name, command, result_path, archive)
                    records[name] = record
                    write_summary(
                        args.output_dir / "summary.csv",
                        records,
                        args.streams,
                        args.repeats,
                    )
                    _warn_point(record)
                _warn_repeats(records, streams, args.repeats)
            write_summary(
                args.output_dir / "summary.csv", records, args.streams, args.repeats
            )
            return int(any(r.get("status") != "ok" for r in records.values()))
