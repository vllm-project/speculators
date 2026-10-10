#!/usr/bin/env python3
"""Throughput vs. interactivity benchmark for a vLLM server (InferenceX style).

Output tokens per second for the GPU against tokens per second for each user, at
N = 1, 2, 4, ... requests kept in flight (closed loop, as InferenceX), on real
prompts. Run it once without a speculator and once with one, and the two curves
show what speculative decoding buys at every concurrency.

    pip install "guidellm>=0.8.0" matplotlib

`collect` runs the sweep with GuideLLM and samples the server's /metrics for
acceptance and queueing, `parse` turns the raw JSONs into a CSV, `validate`
checks the sweep for the usual ways a benchmark lies, and `plot` draws the chart.
The server must be vLLM with prefix caching off; `collect` refuses to start
otherwise, since the prompts repeat.

Per point, over GuideLLM's measurement window (warmup excluded):

    aggregate_output_tps       = output tokens generated in the window / window
                                 (y axis; every overlapping request, prorated)
    interactivity_itl_tps_user = 1000 / mean inter-token latency of the requests
                                 completed in the window (x axis, InferenceX)
    interactivity_tps_user     = aggregate_output_tps / mean requests in flight
                                 (Little's law; `plot --x little`)
    acceptance_length          = 1 + accepted draft tokens / drafts, from vLLM's
                                 counters sampled at the window's edges
    mean_waiting_requests      = vLLM's num_requests_waiting gauge, averaged over
                                 the window

What the chart means, every option, the CSV columns and the validation checks:
docs/user_guide/tutorials/throughput_interactivity.md and
docs/cli/throughput_interactivity.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.metadata
import json
import math
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
except ImportError:  # only `plot` needs it
    plt = Bbox = None  # type: ignore[assignment]

# Thresholds used by `validate`.
STEADY_STATE_TOLERANCE = 0.10  # tokens per completed request vs mean output length
REPEAT_SPREAD_LIMIT = 0.05  # max relative spread of throughput across repeats
QUEUE_TTFT_MS = 1000.0  # a point queues inside the server above this
QUEUE_TTFT_FACTOR = 10.0  # ... or above this multiple of the lowest TTFT
QUEUE_WAITING_MIN = 0.5  # mean requests waiting inside vLLM that count as a queue

# ---------------------------------------------------------------------------
# parse: GuideLLM JSON -> tidy rows
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model",
    "point",
    "streams",
    "repeat",
    "source_json",
    "measured_duration_s",
    "successful_requests",
    "errored_requests",
    "incomplete_requests",
    "measured_requests",
    "completed_rps",
    "aggregate_output_tps",
    "mean_active_concurrency",
    "interactivity_tps_user",
    "mean_itl_ms",
    "interactivity_itl_tps_user",
    "mean_output_tokens",
    "mean_prompt_tokens",
    "mean_output_tokens_per_iteration",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_itl_ms",
    "acceptance_length",
    "num_drafts",
    "num_accepted_tokens",
    "mean_running_requests",
    "mean_waiting_requests",
    "mean_kv_cache_usage",
    "stop_reason",
]

METRICS_SUFFIX = ".metrics.json"


def _mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else float("nan")


def _median(values: list) -> float:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def _percentile(values: list, fraction: float) -> float:
    values = sorted(v for v in values if v is not None)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(fraction * len(values)))]


def _overlap(a: float, b: float, start: float, end: float) -> float:
    return max(0.0, min(b, end) - max(a, start))


def _token_span(request: dict) -> tuple[float, float]:
    """Time span over which a request produced its output tokens."""
    timings = request["info"]["timings"]
    return timings["first_token_iteration"], timings["last_token_iteration"]


def _window_tokens(requests: list, start: float, end: float) -> float:
    """Output tokens generated inside the window, prorated per request."""
    total = 0.0
    for r in requests:
        n = r.get("output_tokens") or 0
        if n <= 0:
            continue
        first, last = _token_span(r)
        if last > first:
            total += n * _overlap(first, last, start, end) / (last - first)
        elif start <= first < end:
            total += n
    return total


def summarize_benchmark(benchmark: dict) -> dict:
    """Reduce one GuideLLM closed-loop benchmark to the metrics the chart needs."""
    strategy = benchmark["config"]["strategy"]
    streams = strategy.get("streams")
    if strategy.get("type_") != "concurrent" or not streams:
        raise ValueError("not a closed-loop (concurrent) benchmark")
    requests = benchmark["requests"]
    successful = requests["successful"] or []
    incomplete = requests.get("incomplete") or []
    errored = requests.get("errored") or []
    timing = benchmark["scheduler_metrics"]
    start = timing["measure_start_time"]
    end = timing["measure_end_time"]
    window = end - start
    if window <= 0:
        raise ValueError("non-positive measurement window")

    # Tokens from every request that overlaps the window, successful or still in
    # flight; in-flight time clipped to the window for every request, errored too.
    every = successful + incomplete + errored
    window_tokens = _window_tokens(successful + incomplete, start, end)
    in_flight = sum(
        _overlap(r["request_start_time"], r["request_end_time"], start, end)
        for r in every
    )
    # Per-request statistics come from the requests that completed inside the
    # window: GuideLLM's `successful` list also holds the ones that finished
    # during warmup, a third of the list at high N.
    measured = [r for r in successful if start <= r["request_end_time"] <= end]
    completed = len(measured)

    aggregate_tps = window_tokens / window
    concurrency = in_flight / window
    constraints = benchmark.get("scheduler_state", {}).get(
        "end_processing_constraints", {}
    )
    output_tokens = sum(r["output_tokens"] or 0 for r in measured)
    itls = [r.get("inter_token_latency_ms") for r in measured]
    itls = [v for v in itls if v]
    mean_itl = _mean(itls)
    ttfts = [r.get("time_to_first_token_ms") for r in measured]
    # Output tokens per streamed chunk: ~1 without speculative decoding, the
    # accepted length + 1 with it (vLLM streams every token a step produced at
    # once). A client-side cross-check for the server's acceptance counters.
    per_iteration = [
        r["output_tokens"] / r["info"]["timings"]["token_iterations"]
        for r in measured
        if r["info"]["timings"]["token_iterations"] and r["output_tokens"]
    ]

    return {
        "point": f"conc{streams:g}",
        "streams": streams,
        "measured_duration_s": window,
        "successful_requests": len(successful),
        # list sizes, not scheduler_metrics.requests_made, which in GuideLLM 0.8
        # also counts requests that were queued but never sent
        "errored_requests": len(errored),
        "incomplete_requests": len(incomplete),
        "measured_requests": completed,
        "completed_rps": completed / window,
        "aggregate_output_tps": aggregate_tps,
        "mean_active_concurrency": concurrency,
        "interactivity_tps_user": (
            aggregate_tps / concurrency if concurrency else float("nan")
        ),
        "mean_itl_ms": mean_itl,
        "interactivity_itl_tps_user": (
            1000.0 / mean_itl if mean_itl and mean_itl > 0 else float("nan")
        ),
        "mean_output_tokens": output_tokens / completed if completed else float("nan"),
        "mean_prompt_tokens": _mean([r.get("prompt_tokens") for r in measured]),
        "mean_output_tokens_per_iteration": _mean(per_iteration),
        "median_ttft_ms": _median(ttfts),
        "p99_ttft_ms": _percentile(ttfts, 0.99),
        "median_itl_ms": _median([r.get("inter_token_latency_ms") for r in measured]),
        "stop_reason": ",".join(constraints.keys()),
    }


SIDECAR_COLUMNS = (
    "acceptance_length",
    "num_drafts",
    "num_accepted_tokens",
    "mean_running_requests",
    "mean_waiting_requests",
    "mean_kv_cache_usage",
)


def read_acceptance(json_path: Path) -> dict:
    """Server metrics `collect` stored next to a run, if any (see `window_metrics`)."""
    sidecar = json_path.with_name(json_path.stem + METRICS_SUFFIX)
    if not sidecar.is_file():
        return dict.fromkeys(SIDECAR_COLUMNS, float("nan"))
    payload = json.loads(sidecar.read_text())
    delta = payload.get("delta") or {}
    gauges = payload.get("gauges") or {}

    def gauge_mean(name: str) -> float:
        return (gauges.get(name) or {}).get("mean", float("nan"))

    return {
        "acceptance_length": delta.get("acceptance_length", float("nan")),
        "num_drafts": delta.get("num_drafts", float("nan")),
        "num_accepted_tokens": delta.get("num_accepted_tokens", float("nan")),
        "mean_running_requests": gauge_mean("num_requests_running"),
        "mean_waiting_requests": gauge_mean("num_requests_waiting"),
        "mean_kv_cache_usage": gauge_mean("kv_cache_usage_perc"),
    }


def _repeat_of(stem: str) -> int:
    """`<point>_r<N>` -> N; anything else -> 1."""
    _, _, tail = stem.rpartition("_r")
    return int(tail) if tail.isdigit() else 1


def parse_dir(json_dir: Path, label: str) -> list[dict]:
    """Summarize every GuideLLM benchmark in every `*.json` under `json_dir`.

    Points are named `conc<N>` from the benchmark's own strategy; the repeat comes
    from file names of the form `<point>_r<N>.json` (what `collect` writes).
    """
    rows = []
    for path in sorted(json_dir.rglob("*.json")):
        if path.name.endswith(METRICS_SUFFIX):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        for index, benchmark in enumerate(payload.get("benchmarks") or []):
            try:
                summary = summarize_benchmark(benchmark)
            except (KeyError, ValueError) as exc:
                print(f"  skip {path.name}[{index}]: {exc}", file=sys.stderr)
                continue
            summary.update(read_acceptance(path))
            summary.update(
                model=label, repeat=_repeat_of(path.stem), source_json=path.name
            )
            rows.append(summary)
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# server metrics: acceptance and scheduler gauges from vLLM's /metrics
# ---------------------------------------------------------------------------

METRICS_INTERVAL_S = 2.0  # seconds between /metrics samples during a run
SPEC_PREFIX = "vllm:spec_decode_"
# Prompt tokens looked up in the server's prefix cache: any movement means prefix
# caching is on (older vLLM names it gpu_prefix_cache_queries).
PREFIX_CACHE_QUERIES = ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries")
# Scheduler gauges sampled during a run (older vLLM names the cache one gpu_*).
GAUGES = {
    "num_requests_running": ("vllm:num_requests_running",),
    "num_requests_waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage_perc": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
NAMED_SERIES = {"prefix_cache_queries": PREFIX_CACHE_QUERIES, **GAUGES}
_RE_METRIC = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)$")
_RE_POSITION = re.compile(r'position="(\d+)"')
_RE_CACHE_CONFIG = re.compile(r"^vllm:cache_config_info\{([^}]*)\}")
_RE_PREFIX_CACHING = re.compile(r'enable_prefix_caching="([^"]*)"')


def fetch_text(url: str, timeout: float = 30.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def spec_decode_counters(text: str) -> dict:
    """Sum vLLM's speculative-decoding counters and scheduler gauges over engines.

    Reads `vllm:spec_decode_num_drafts`, `..._num_draft_tokens`,
    `..._num_accepted_tokens` and `..._num_accepted_tokens_per_pos` (with or
    without the `_total` suffix) and returns the sums plus the per-position list,
    `vllm:prefix_cache_queries` (which must stay at 0, see
    `prefix_caching_enabled`), and the gauges in GAUGES (None when the server
    does not export them). The `<counter>_created` timestamps prometheus_client
    adds are skipped.
    """
    sums: dict[str, float] = defaultdict(float)
    per_position: dict[int, float] = defaultdict(float)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_METRIC.match(line)
        if not match:
            continue
        name = match.group(1)
        if name.endswith("_created"):  # a counter's creation time, not a count
            continue
        name = name.replace("_total", "")
        labels, value = match.group(2) or "", float(match.group(3))
        for key, names in NAMED_SERIES.items():
            if name in names:
                sums[key] += value
        if not name.startswith(SPEC_PREFIX):
            continue
        if "per_pos" in name:
            position = _RE_POSITION.search(labels)
            if position:
                per_position[int(position.group(1))] += value
        else:
            sums[name[len(SPEC_PREFIX) :]] += value
    positions = (
        [per_position[i] for i in range(max(per_position) + 1)] if per_position else []
    )
    return {
        "num_drafts": sums.get("num_drafts", 0.0),
        "num_draft_tokens": sums.get("num_draft_tokens", 0.0),
        "num_accepted_tokens": sums.get("num_accepted_tokens", 0.0),
        "accepted_per_position": positions,
        "prefix_cache_queries": sums.get("prefix_cache_queries", 0.0),
        **{key: sums.get(key) for key in GAUGES},
    }


def prefix_caching_enabled(text: str) -> bool | None:
    """Whether the vLLM server behind this /metrics page has prefix caching on.

    Read from the `enable_prefix_caching` label of `vllm:cache_config_info`; if the
    server does not export it, from whether the prefix-cache counters have ever
    moved. None when neither is there (not a vLLM server).
    """
    for raw_line in text.splitlines():
        match = _RE_CACHE_CONFIG.match(raw_line.strip())
        if match:
            flag = _RE_PREFIX_CACHING.search(match.group(1))
            if flag:
                return flag.group(1).strip().lower() == "true"
    if spec_decode_counters(text)["prefix_cache_queries"] > 0:
        return True
    return None


def acceptance_delta(before: dict, after: dict) -> dict:
    """Acceptance over one run, from two snapshots of the counters."""
    queries = after.get("prefix_cache_queries", 0.0) - before.get(
        "prefix_cache_queries", 0.0
    )
    drafts = after["num_drafts"] - before["num_drafts"]
    draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    per_position = [
        a - b
        for a, b in zip_longest(
            after["accepted_per_position"],
            before["accepted_per_position"],
            fillvalue=0.0,
        )
    ]
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        # the same definition as scripts/evaluate/perf_utils.py: drafted tokens
        # accepted per draft, plus the one token the verifier always emits
        "acceptance_length": 1 + accepted / drafts if drafts > 0 else None,
        "acceptance_at_position": (
            [count / drafts for count in per_position] if drafts > 0 else []
        ),
        # prompt tokens looked up in the prefix cache: above 0 means it was on
        "prefix_cache_queries": queries,
    }


def window_metrics(
    samples: list[tuple[float, dict]], start: float | None, end: float | None
) -> dict | None:
    """What `collect` stores in a run's `.metrics.json`, from timed /metrics samples.

    Counters are differenced between the samples nearest the measurement window's
    start and end, so warmup and GuideLLM's startup are not in the acceptance
    numbers (the whole-run difference is kept under `whole_run`). The gauges are
    averaged over the samples inside the window. Without a window the whole run
    is used.
    """
    if not samples:
        return None
    samples = sorted(samples, key=lambda s: s[0])
    (first_time, first), (last_time, last) = samples[0], samples[-1]
    if start is None or end is None or end <= start:
        start, end = first_time, last_time
        before_time, before, after_time, after = first_time, first, last_time, last
    else:
        before_time, before = min(samples, key=lambda s: abs(s[0] - start))
        after_time, after = min(samples, key=lambda s: abs(s[0] - end))
    inside = [s for s in samples if start <= s[0] <= end] or [
        (before_time, before),
        (after_time, after),
    ]
    gauges: dict[str, dict | None] = {}
    for key in GAUGES:
        values = [c.get(key) for _, c in inside]
        values = [v for v in values if v is not None]
        gauges[key] = (
            {"mean": statistics.fmean(values), "max": max(values), "n": len(values)}
            if values
            else None
        )
    return {
        "window": {
            "start": start,
            "end": end,
            "before_sample": before_time,
            "after_sample": after_time,
        },
        "before": before,
        "after": after,
        "delta": acceptance_delta(before, after),
        "gauges": gauges,
        "whole_run": {
            "before_sample": first_time,
            "after_sample": last_time,
            "delta": acceptance_delta(first, last),
        },
        "samples": len(samples),
        # [time, running, waiting, kv cache usage] per sample, for a closer look
        "series": [[round(t, 3), *(c.get(key) for key in GAUGES)] for t, c in samples],
    }


class MetricsSampler:
    """Poll a /metrics endpoint from a thread while a GuideLLM run is in progress."""

    def __init__(self, url: str, interval: float = METRICS_INTERVAL_S):
        self.url = url
        self.interval = interval
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        text = fetch_text(self.url, timeout=min(10.0, max(1.0, self.interval)))
        if text is not None:
            self.samples.append((time.time(), spec_decode_counters(text)))

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> MetricsSampler:
        self._thread.start()
        return self

    def stop(self) -> list[tuple[float, dict]]:
        self._stop.set()
        self._thread.join(timeout=30)
        self._sample()  # one last snapshot after the run
        return self.samples


def _measurement_window(json_path: Path) -> tuple[float | None, float | None]:
    """GuideLLM's measurement window from the JSON it wrote, if readable."""
    try:
        timing = json.loads(json_path.read_text())["benchmarks"][0]["scheduler_metrics"]
        return float(timing["measure_start_time"]), float(timing["measure_end_time"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None, None


def refuse_prefix_caching(metrics_url: str) -> None:
    """Exit unless the vLLM server behind `metrics_url` says prefix caching is off."""
    text = fetch_text(metrics_url, timeout=10)
    if text is None:
        raise SystemExit(
            f"no /metrics at {metrics_url}: this benchmark needs vLLM's metrics "
            "(acceptance counters, and the check that prefix caching is off)"
        )
    enabled = prefix_caching_enabled(text)
    if enabled is None:
        raise SystemExit(
            f"{metrics_url} does not report enable_prefix_caching; is this a vLLM "
            "server?"
        )
    if enabled:
        raise SystemExit(
            f"{metrics_url} reports prefix caching enabled. The prompts repeat, so "
            "cached prefill would make every number a cached-workload number; serve "
            "with --no-enable-prefix-caching."
        )


# ---------------------------------------------------------------------------
# collect: drive GuideLLM
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "RedHatAI/speculator_benchmarks"
HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/main/{subset}.jsonl"
MIN_DATASET_REQUESTS = 5000
REQUESTS_PER_STREAM = 20
COLUMN_MAPPER = "kind=generative_column_mapper,column_mappings.text_column=prompt"


def _dataset_rows(dataset: str, subset: str) -> list:
    """Rows of `<subset>.jsonl` from a local file or directory, or a HF dataset id."""
    local = Path(dataset)
    for candidate in (local / f"{subset}.jsonl", local):
        if candidate.is_file():
            lines = candidate.read_text().splitlines()
            return [json.loads(line) for line in lines if line.strip()]
    if local.exists():
        raise FileNotFoundError(f"{dataset}: no {subset}.jsonl inside, and not a file")
    url = HF_RESOLVE.format(dataset=dataset, subset=subset)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def materialize_dataset(args: argparse.Namespace, out_dir: Path) -> Path:
    """Write `<out_dir>/_data/<subset>_x<K>.jsonl`: the subset repeated and shuffled.

    GuideLLM stops a benchmark when its dataset runs out (`requests_exhausted`),
    and the speculator_benchmarks subsets hold 80-200 rows, so a 128-stream run
    would end within seconds. K defaults to enough rows for 20 requests per stream
    and at least 5,000 requests. The shuffle is seeded, so two sweeps see the same
    order.
    """
    rows = _dataset_rows(args.dataset, args.subset)
    if not rows:
        raise ValueError(f"{args.dataset}/{args.subset}: no rows")
    if args.prompt_column not in rows[0]:
        raise ValueError(
            f"column {args.prompt_column!r} not in {sorted(rows[0])}; "
            "use --prompt-column"
        )
    repeat = args.dataset_repeat
    if repeat <= 0:
        target = max(MIN_DATASET_REQUESTS, REQUESTS_PER_STREAM * max(args.streams))
        repeat = max(1, math.ceil(target / len(rows)))
    prompts = [{"prompt": r[args.prompt_column]} for r in rows] * repeat
    random.Random(0).shuffle(prompts)
    path = out_dir / "_data" / f"{Path(args.subset).stem}_x{repeat}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or args.overwrite:
        with path.open("w") as handle:
            for row in prompts:
                handle.write(json.dumps(row) + "\n")
    print(f"dataset: {len(rows)} rows x {repeat} = {len(prompts)} prompts -> {path}")
    return path


def normalize_target(target: str) -> str:
    """Server base URL without a trailing slash or /v1 (GuideLLM adds the route)."""
    target = target.rstrip("/")
    if target.endswith("/v1"):
        target = target[: -len("/v1")]
    return target


def build_backend(args: argparse.Namespace) -> str:
    """GuideLLM backend spec as JSON, so nested `extras` survive."""
    backend: dict[str, Any] = {
        "kind": "openai_http",
        "target": args.target,
        "model": args.model,
        "request_format": "/v1/chat/completions",
        "max_tokens": args.max_tokens,
    }
    body: dict[str, Any] = {}
    for item in args.extra_body:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--extra-body expects KEY=JSON, got {item!r}")
        try:
            body[key] = json.loads(value)
        except json.JSONDecodeError:
            body[key] = value  # a bare string
    if body:
        backend["extras"] = {"body": body}
    return json.dumps(backend)


def _git_sha(path: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603
        [git, "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def write_provenance(out_dir: Path, args: argparse.Namespace) -> None:
    """Append a block to `<out_dir>/bench_command.txt`: enough to rerun and cite.

    One block per `collect` invocation, since a sweep is often built up by
    several calls into one directory (short and long windows, added points).
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"timestamp: {now}",
        f"command: {shlex.join([sys.executable, *sys.argv])}",
        f"cwd: {Path.cwd()}",
        f"python: {sys.version.split()[0]}",
        f"git_sha: {_git_sha(Path(__file__).resolve().parent) or 'unknown'}",
    ]
    for package in ("guidellm", "vllm", "speculators"):
        try:
            lines.append(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{package}: not installed")
    for name, path in (("server_version", "/version"), ("server_models", "/v1/models")):
        text = fetch_text(args.target + path, timeout=10)
        lines.append(f"{name}: {' '.join(text.split()) if text else 'unavailable'}")
    path = out_dir / "bench_command.txt"
    existing = path.read_text() if path.exists() else ""
    separator = "\n" if existing and not existing.endswith("\n\n") else ""
    path.write_text(existing + separator + "\n".join(lines) + "\n")


def _run_point(command: list[str], target_json: Path, metrics_url: str) -> bool:
    """Run one GuideLLM invocation with /metrics sampled, and write its sidecar.

    False when GuideLLM failed, or the server's prefix-cache counters moved during
    the run (prefix caching is on after all). See `window_metrics` for the sidecar.
    """
    sidecar = target_json.with_name(target_json.stem + METRICS_SUFFIX)
    sidecar.unlink(missing_ok=True)  # never merge an earlier run's metrics
    sampler = MetricsSampler(metrics_url).start()
    returncode = subprocess.run(command, check=False).returncode  # noqa: S603
    samples = sampler.stop()
    if returncode != 0:
        print(
            f"!! guidellm exited with {returncode}: {target_json.name}", file=sys.stderr
        )
        return False
    metrics = window_metrics(samples, *_measurement_window(target_json))
    if metrics is None:
        print(
            f"!! no /metrics at {metrics_url} during {target_json.name}",
            file=sys.stderr,
        )
        return False
    sidecar.write_text(json.dumps({"metrics_url": metrics_url, **metrics}, indent=1))
    if metrics["whole_run"]["delta"]["prefix_cache_queries"] > 0:
        print(
            f"!! {target_json.name}: the server's prefix-cache counters moved during "
            "the run, so prefix caching is on; serve with --no-enable-prefix-caching "
            "and re-run this point",
            file=sys.stderr,
        )
        return False
    return True


def _check_collect_args(args: argparse.Namespace) -> None:
    if any(n <= 0 for n in args.streams):
        raise SystemExit("--streams must be positive")
    if 0 < args.warmup_seconds < 1:
        raise SystemExit(
            "--warmup-seconds must be 0 or at least 1: GuideLLM reads a value "
            "below 1 as a fraction of the run"
        )
    if args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")


def _write_results(out_dir: Path, args: argparse.Namespace, failures: int) -> int:
    rows = parse_dir(out_dir, args.label)
    if not rows:
        print("no benchmarks parsed", file=sys.stderr)
        return 1
    csv_path = Path(args.csv) if args.csv else out_dir / f"{args.label}.csv"
    write_csv(rows, csv_path)
    print(f"\nwrote {len(rows)} rows -> {csv_path}")
    validate_rows(rows)
    if failures:
        print(
            f"!! {failures} run(s) failed; their points are missing from {csv_path}",
            file=sys.stderr,
        )
    return 1 if failures else 0


def collect(args: argparse.Namespace) -> int:
    """Run one GuideLLM invocation per (N, repeat), then parse and validate.

    Existing JSONs are skipped unless --overwrite, so a sweep can be resumed or
    extended into the same directory. The server must have prefix caching off:
    `collect` reads the setting from /metrics before starting and refuses to run
    while it is on.
    """
    _check_collect_args(args)
    guidellm = shutil.which(args.guidellm_bin)
    if guidellm is None and not args.dry_run:
        raise SystemExit(
            f"{args.guidellm_bin!r} not found on PATH; pip install guidellm"
        )
    args.target = normalize_target(args.target)
    metrics_url = args.target + "/metrics"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = materialize_dataset(args, out_dir)
    if not args.dry_run:
        refuse_prefix_caching(metrics_url)
        write_provenance(out_dir, args)
    base_command = [
        guidellm or args.guidellm_bin,
        "run",
        "--backend",
        build_backend(args),
        "--tokenizer",
        f"kind=hf_auto,model={args.tokenizer or args.model}",
        "--data",
        f"kind=json_file,path={data_path},load_kwargs.split=train",
        "--data-column-mapper",
        COLUMN_MAPPER,
        "--constraint",
        f"kind=max_duration,seconds={args.max_seconds + args.warmup_seconds:g}",
        "--metrics",
        "kind=generative,sample_size=0",
        "--disable-console-interactive",
    ]
    warmup = f",warmup={args.warmup_seconds:g}" if args.warmup_seconds else ""

    failures = 0
    for streams in args.streams:
        for repeat in range(1, args.repeats + 1):
            target_json = out_dir / f"conc{streams:g}_r{repeat}.json"
            if target_json.exists() and not args.overwrite:
                print(f"exists, skipping: {target_json.name}")
                continue
            command = [
                *base_command,
                "--profile",
                f"kind=concurrent,streams={streams:g}{warmup}",
                "--output",
                f"kind=json,path={target_json}",
            ]
            print(f"\n=== {target_json.stem}\n{shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            if not _run_point(command, target_json, metrics_url):
                failures += 1
                if not args.keep_going:
                    return 1

    if args.dry_run:
        print("\n(dry run, nothing executed)")
        return 0
    return _write_results(out_dir, args, failures)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [r for r in csv.DictReader(handle) if any(v for v in r.values())]


def _group_mean(group: list[dict], field: str) -> float | None:
    values = [_as_float(r.get(field)) for r in group]
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


AGGREGATE_FIELDS = {
    "x_itl": "interactivity_itl_tps_user",
    "x_little": "interactivity_tps_user",
    "y": "aggregate_output_tps",
    "completed_rps": "completed_rps",
    "streams": "streams",
    "conc": "mean_active_concurrency",
    "itl_mean": "mean_itl_ms",
    "ttft": "median_ttft_ms",
    "tokens": "mean_output_tokens",
    "acceptance": "acceptance_length",
    "running": "mean_running_requests",
    "waiting": "mean_waiting_requests",
}


def aggregate_points(rows: list[dict]) -> list[dict]:
    """Average repeats of the same point; carry the per-repeat spread."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["point"]].append(row)

    points = []
    for name, group in groups.items():
        throughputs = [_as_float(r.get("aggregate_output_tps")) for r in group]
        throughputs = [t for t in throughputs if t is not None]
        spread = (
            (max(throughputs) - min(throughputs)) / statistics.fmean(throughputs)
            if len(throughputs) > 1 and statistics.fmean(throughputs)
            else 0.0
        )
        point = {
            key: _group_mean(group, field) for key, field in AGGREGATE_FIELDS.items()
        }
        point.update(
            point=name,
            repeats=len(group),
            spread=spread,
            stop_reason=group[0].get("stop_reason") or "",
        )
        points.append(point)
    points.sort(key=lambda p: p["streams"] or 0.0)
    return points


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def _print_table(points: list[dict]) -> None:
    header = (
        "point",
        "N",
        "tok/s",
        "conc",
        "1/itl",
        "little",
        "itl_ms",
        "ttft_ms",
        "waiting",
        "accept",
        "spread",
    )
    layout = "{:>8} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>6} {:>6}"
    print(
        "\nper point, mean over repeats: N = streams; tok/s = output throughput; "
        "conc = mean requests in flight; 1/itl = 1000 / mean ITL (tok/s per user, "
        "the x axis); little = throughput / conc; waiting = mean requests queued "
        "inside vLLM; accept = acceptance length; spread = throughput range across "
        "repeats"
    )
    print(layout.format(*header))
    for p in points:
        print(
            layout.format(
                p["point"],
                _fmt(p["streams"], ".0f"),
                _fmt(p["y"], ".0f"),
                _fmt(p["conc"], ".1f"),
                _fmt(p["x_itl"], ".1f"),
                _fmt(p["x_little"], ".1f"),
                _fmt(p["itl_mean"], ".2f"),
                _fmt(p["ttft"], ".0f"),
                _fmt(p["waiting"], ".1f"),
                _fmt(p["acceptance"], ".2f"),
                f"{100 * p['spread']:.1f}%",
            )
        )


def _queue_label(p: dict) -> str:
    text = f"{p['point']} ({p['ttft'] / 1000:.1f} s"
    if p["waiting"] is not None:
        text += f", {p['waiting']:.0f} waiting, {p['running'] or 0:.0f} running"
    return text + ")"


def _check_exhausted(points: list[dict]) -> list[str]:
    exhausted = [p["point"] for p in points if "requests_exhausted" in p["stop_reason"]]
    if not exhausted:
        return []
    return [
        f"{len(exhausted)} points ended on requests_exhausted "
        f"({', '.join(exhausted)}): the sweep ran out of dataset, so its concurrency "
        "is bounded by dataset size, not the server. Set --dataset-repeat above the "
        "automatic choice."
    ]


def _check_steady_state(points: list[dict]) -> list[str]:
    """Tokens generated in the window per completed request vs the output length."""
    for p in points:
        if not (p["y"] and p["completed_rps"] and p["tokens"]):
            continue
        accounted = p["y"] / p["completed_rps"]
        if abs(accounted - p["tokens"]) > STEADY_STATE_TOLERANCE * p["tokens"]:
            return [
                f"{p['point']}: the window holds {accounted:.0f} generated tokens per "
                f"completed request but requests average {p['tokens']:.0f}; the run "
                "did not reach steady state within the window. Lengthen --max-seconds "
                "and --warmup-seconds."
            ]
    return []


def _check_repeat_noise(points: list[dict]) -> list[str]:
    noisy = [
        p for p in points if p["repeats"] > 1 and p["spread"] > REPEAT_SPREAD_LIMIT
    ]
    if not noisy:
        return []
    worst = max(noisy, key=lambda p: p["spread"])
    return [
        f"{len(noisy)} points vary more than {100 * REPEAT_SPREAD_LIMIT:.0f}% across "
        f"repeats (worst {worst['spread'] * 100:.1f}% at {worst['point']}): treat "
        "differences smaller than that as noise."
    ]


def _check_queueing(points: list[dict]) -> list[str]:
    """Past what the server can hold, queueing shows up as TTFT. vLLM's waiting
    gauge says whether the queue is inside the server; a starved API server or
    client raises TTFT the same way."""
    ttfts = [p["ttft"] for p in points if p["ttft"]]
    if not ttfts:
        return []
    floor = max(QUEUE_TTFT_MS, QUEUE_TTFT_FACTOR * min(ttfts))
    queued = [p for p in points if p["ttft"] and p["ttft"] > floor]
    in_server = [
        p for p in queued if p["waiting"] is None or p["waiting"] >= QUEUE_WAITING_MIN
    ]
    elsewhere = [p for p in queued if p not in in_server]
    warnings = []
    if in_server:
        names = ", ".join(_queue_label(p) for p in in_server)
        warnings.append(
            f"{len(in_server)} points have a median TTFT above {floor / 1000:.1f} s: "
            f"{names}. More streams than the server can hold at once wait inside it "
            "(a batch cap, or a full KV cache), so these points measure queueing plus "
            "decode. Raise max-num-seqs, free KV cache memory, or stop the sweep below "
            "this N."
        )
    if elsewhere:
        names = ", ".join(_queue_label(p) for p in elsewhere)
        warnings.append(
            f"{len(elsewhere)} points have a median TTFT above {floor / 1000:.1f} s "
            f"while vLLM reports no waiting requests: {names}. The delay is outside "
            "the scheduler: a starved API server, the GuideLLM client, or the "
            "network. Check the host's load before blaming the server."
        )
    return warnings


CHECKS = (_check_exhausted, _check_steady_state, _check_repeat_noise, _check_queueing)


def validate_rows(rows: list[dict]) -> list[str]:
    points = aggregate_points(rows)
    _print_table(points)
    warnings = [warning for check in CHECKS for warning in check(points)]
    print()
    for warning in warnings:
        print(f"WARNING: {warning}\n")
    if not warnings:
        print("no warnings\n")
    return warnings


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

PALETTE = ["#b52513", "#2c6e9b", "#7a8c3f", "#b8b1a8", "#8a5a9e"]
BG, GRID, INK, MUTED = "#f2f2f2", "#cfdae2", "#181818", "#666666"
YLABEL = "Output Token Throughput (tok/s)"
XLABELS = {
    "itl": "Interactivity (tok/s/user, 1 / mean inter-token latency)",
    "little": "Interactivity (tok/s/user, throughput / requests in flight)",
}
# label slots tried around a marker, in points; nearest first
LABEL_OFFSETS = [
    (8, 6),
    (8, -16),
    (-8, 6),
    (-8, -16),
    (8, 22),
    (-8, 22),
    (8, -32),
    (-8, -32),
]
MARKER_PX = 8  # markers closer than this sit on top of each other


def _load_series(args: argparse.Namespace) -> list[dict]:
    series = []
    for spec in args.series:
        # CSV[:NAME[:#COLOR]]: the color is a trailing `:#...` field, so the name
        # may contain colons ("B200: no speculator").
        path_text, _, name = spec.partition(":")
        color = ""
        head, sep, tail = name.rpartition(":")
        if sep and tail.startswith("#"):
            name, color = head, tail
        path = Path(path_text)
        if not path.is_file():
            raise SystemExit(
                f"{path}: no such CSV (did every point of its sweep fail?)"
            )
        points = aggregate_points(load_rows(path))
        for p in points:
            p["x"] = p["x_itl"] if args.x == "itl" else p["x_little"]
        points = [p for p in points if p["x"] and p["y"]]
        if not points:
            raise SystemExit(f"no points with an x and y value in {path}")
        series.append(
            {
                "name": name or path.stem,
                "color": color or PALETTE[len(series) % len(PALETTE)],
                "points": points,  # in N order, from aggregate_points
            }
        )
    return series


def _label_points(ax: Any, points: list[dict], taken: list[Any]) -> None:
    """Write `N=...` next to each marker, in the first slot that overlaps nothing.

    Markers drawn on top of each other (a saturated server puts N = 32, 64 and
    128 on one spot) share one label. `taken` holds the boxes already used on the
    canvas, markers included, and grows with every label placed.
    """
    clusters: list[tuple[list[dict], tuple[float, float]]] = []
    for p in points:
        px, py = ax.transData.transform((p["x"], p["y"]))
        if clusters and all(
            abs(v - w) < MARKER_PX
            for v, w in zip((px, py), clusters[-1][1], strict=True)
        ):
            clusters[-1][0].append(p)
        else:
            clusters.append(([p], (px, py)))
    for _, (px, py) in clusters:
        taken.append(Bbox.from_extents(px - 7, py - 7, px + 7, py + 7))
    for cluster, _ in clusters:
        first, last = cluster[0]["streams"], cluster[-1]["streams"]
        label = f"N={first:.0f}" if len(cluster) == 1 else f"N={first:.0f}..{last:.0f}"
        for dx, dy in LABEL_OFFSETS:
            text = ax.annotate(
                label,
                (cluster[0]["x"], cluster[0]["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else "right",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
            box = text.get_window_extent()
            inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(
                box.x1, box.y1
            )
            if inside and not any(box.overlaps(other) for other in taken):
                taken.append(box)
                break
            text.remove()


def plot(args: argparse.Namespace) -> int:
    if plt is None:
        raise SystemExit("plot needs matplotlib: pip install matplotlib")
    series = _load_series(args)
    every = [p for s in series for p in s["points"]]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=135, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1.08 * max(p["x"] for p in every))
    ax.set_ylim(0, 1.08 * max(p["y"] for p in every))
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(
        lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"  # noqa: PLR2004
    )
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in series:
        xs, ys = [p["x"] for p in s["points"]], [p["y"] for p in s["points"]]
        ax.plot(
            xs,
            ys,
            color=s["color"],
            linewidth=3,
            marker="o",
            markersize=7,
            label=s["name"],
        )
    ax.set_xlabel(XLABELS[args.x], fontsize=12, color=INK)
    ax.set_ylabel(YLABEL, fontsize=12, color=INK)
    fig.suptitle(
        args.title, x=0.05, ha="left", fontsize=22, fontweight="bold", color=INK
    )
    if args.subtitle:
        ax.set_title(
            args.subtitle, loc="left", fontsize=11, color=MUTED, pad=14, wrap=True
        )
    if len(series) > 1:
        ax.legend(loc="upper right", fontsize=11, facecolor="white", framealpha=1)
    if args.note:
        fig.text(0.05, 0.015, args.note, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    taken: list[Any] = []  # labels of later series avoid those of earlier ones
    for s in series:
        _label_points(ax, s["points"], taken)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def run_parse(args: argparse.Namespace) -> int:
    rows = parse_dir(Path(args.json_dir), args.label)
    if not rows:
        print(f"no benchmarks found under {args.json_dir}", file=sys.stderr)
        return 1
    write_csv(rows, Path(args.csv))
    print(f"wrote {len(rows)} rows -> {args.csv}")
    validate_rows(rows)
    return 0


def run_validate(args: argparse.Namespace) -> int:
    validate_rows(load_rows(Path(args.csv)))
    return 0


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def _add_collect_parser(sub: Any) -> None:
    c = sub.add_parser("collect", help="run a GuideLLM sweep, then parse + validate")
    c.add_argument(
        "--target",
        required=True,
        help="vLLM server base URL without /v1, e.g. http://127.0.0.1:8000 (a "
        "trailing /v1 is removed); /metrics must be served there too",
    )
    c.add_argument(
        "--model",
        required=True,
        help="model name as the server reports it (the request's `model` field)",
    )
    c.add_argument(
        "--tokenizer", default="", help="tokenizer path or HF id (default: --model)"
    )
    c.add_argument(
        "--extra-body",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="extra field sent in every request body, repeatable, e.g. temperature=0 "
        "for greedy decoding or top_p=0.95; without it the server's sampling defaults "
        "apply (vLLM takes them from the model's generation_config.json)",
    )
    load = c.add_argument_group("load (closed loop, InferenceX style)")
    load.add_argument(
        "--streams",
        type=_int_list,
        required=True,
        help="requests kept in flight at all times; one point per value, e.g. "
        "1,2,4,8,16,32,64,128",
    )
    load.add_argument("--repeats", type=int, default=3, help="runs per point")
    load.add_argument(
        "--max-seconds",
        type=float,
        default=100.0,
        help="measurement window per point, in seconds, after the warmup",
    )
    load.add_argument(
        "--warmup-seconds",
        type=float,
        default=30.0,
        help="warmup per point, in seconds (0, or at least 1), excluded from every "
        "number",
    )
    data = c.add_argument_group("data (real prompts through /v1/chat/completions)")
    data.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset id, or a local dir/jsonl (default: {DEFAULT_DATASET})",
    )
    data.add_argument(
        "--subset",
        required=True,
        help="file name without .jsonl, e.g. HumanEval or math_reasoning",
    )
    data.add_argument(
        "--prompt-column", default="prompt", help="dataset column with the prompt"
    )
    data.add_argument(
        "--dataset-repeat",
        type=int,
        default=0,
        help="repeat the dataset this many times (0 = enough for the sweep)",
    )
    data.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="max_tokens sent with every request, e.g. 1024",
    )
    c.add_argument("--out-dir", required=True)
    c.add_argument(
        "--label",
        required=True,
        help="name of this server configuration, written to the CSV's model column "
        "(e.g. baseline, dspark)",
    )
    c.add_argument("--csv", default="", help="default: <out-dir>/<label>.csv")
    c.add_argument("--guidellm-bin", default="guidellm", help="GuideLLM executable")
    c.add_argument(
        "--dry-run", action="store_true", help="print commands only; do this first"
    )
    c.add_argument("--overwrite", action="store_true", help="re-run existing points")
    c.add_argument(
        "--keep-going",
        action="store_true",
        help="run the remaining points after a failed one; the exit status still "
        "reports the failure",
    )
    c.set_defaults(func=collect)


def _add_plot_parser(sub: Any) -> None:
    g = sub.add_parser("plot", help="render the chart")
    g.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="CSV[:NAME[:#COLOR]]",
        help="one curve per CSV, fields separated by colons; repeatable; later "
        "series are drawn over earlier ones",
    )
    g.add_argument("--out", required=True)
    g.add_argument("--title", default="Output Throughput vs. Interactivity")
    g.add_argument("--subtitle", default="")
    g.add_argument(
        "--x",
        choices=["itl", "little"],
        default="itl",
        help="x axis: 1000/mean ITL (InferenceX, default) or throughput/in-flight",
    )
    g.add_argument("--note", default="", help="one grey line under the x-axis label")
    g.set_defaults(func=plot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_collect_parser(sub)

    p = sub.add_parser("parse", help="GuideLLM JSONs -> CSV")
    p.add_argument("json_dir")
    p.add_argument("--label", required=True)
    p.add_argument("--csv", required=True)
    p.set_defaults(func=run_parse)

    v = sub.add_parser("validate", help="sanity-check a CSV")
    v.add_argument("csv")
    v.set_defaults(func=run_validate)

    _add_plot_parser(sub)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
