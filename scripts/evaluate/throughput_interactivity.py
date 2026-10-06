#!/usr/bin/env python3
"""Throughput vs. interactivity benchmark for a vLLM server (InferenceX style).

Measures how many output tokens a server produces per second at increasing load,
and how fast each user's stream is at that load, and draws both on one chart: the
chart SemiAnalysis publishes on InferenceX (https://inferencex.semianalysis.com).
Run it once without a speculator and once with one, and the two curves show what
speculative decoding buys at every concurrency.

Self-contained helper: one file, Python 3.10+, standard library only except
GuideLLM (>= 0.7.1, called as a shell command by `collect`) and Pillow (`plot`).

    pip install "guidellm>=0.8.0" pillow


WHAT THE CHART SHOWS
--------------------
One point per load level. For each point, over GuideLLM's measurement window
[scheduler_metrics.measure_start_time, measure_end_time]:

    aggregate_output_tps       = output tokens generated in the window / window
                                 (y axis)
    mean_itl_ms                = mean over successful requests of
                                 (last token - first token) / (output tokens - 1)
    interactivity_itl_tps_user = 1000 / mean_itl_ms   (x axis, InferenceX)
    mean_active_concurrency    = sum(per-request time in flight, clipped to the
                                 window) / window
    interactivity_tps_user     = aggregate_output_tps / mean_active_concurrency
                                 (x axis, Little's law; `plot --x little`)
    achieved_rps               = requests started in the window / window
    completed_rps              = requests completed in the window / window
    acceptance_length          = 1 + accepted draft tokens / drafts, from the
                                 server's /metrics counters (speculators only)

The default x axis is the InferenceX definition: 1000 / mean inter-token latency,
the decode speed one user sees once tokens are flowing. GuideLLM's
`inter_token_latency_ms` is the same formula as InferenceX's TPOT,
(end-to-end - TTFT) / (tokens - 1), so the curves are comparable with the
InferenceX dashboard. With speculative decoding, a step emits several tokens, so
ITL per token drops while the step time rises; that is the effect to measure.
The Little's-law column divides tokens by the whole time a request was in flight,
so queue wait and TTFT count against the user. Below capacity the two agree
within a few percent; past capacity only the Little's-law one drops.

Tokens and in-flight time are counted over every request that overlaps the
window, successful or still in flight when the run stopped. A request's output
tokens are spread evenly between its first and last token and only the part
inside the window counts, so requests that started during warmup or were cut off
at the end are neither over- nor under-counted.


TWO WAYS TO LOAD THE SERVER
---------------------------
--streams N,N,...   closed loop (what InferenceX does): N requests in flight at
                    all times; each stream sends its next request the moment the
                    previous one completes. Concurrency is the knob. The server
                    is never over-queued, so the sweep can go as deep into
                    saturation as max-num-seqs allows. Points: conc<N>.
--rates r,r,...     open loop: requests arrive every 1/r seconds whatever the
                    server does. Concurrency is an outcome (r x latency), and
                    past capacity the queue grows without bound. Points: rate<r>.

Use --streams to reproduce an InferenceX-style curve (N doubling from 1). Use
--rates when the question is "what arrival rate can this server absorb".


DATA
----
--dataset ID|DIR --subset NAME [--prompt-column prompt] [--dataset-repeat K]
                    real prompts, e.g. RedHatAI/speculator_benchmarks with
                    --subset HumanEval. The jsonl is fetched once, repeated K
                    times (default: enough for the sweep) and shuffled into
                    <out-dir>/_data, because GuideLLM stops when a dataset runs
                    out. Sent through /v1/chat/completions. Set --max-tokens; add
                    --ignore-eos to force that length. Use real data to measure
                    a speculator: a drafter has nothing to predict in random text.
--prompt-tokens P --output-tokens O [--range-ratio 0.8]
                    random text; each request's lengths are drawn uniformly
                    from [ratio x L, L]. InferenceX uses 0.8. The jitter is not
                    cosmetic: with fixed lengths, closed-loop streams finish
                    together, restart together and prefill in one burst, which
                    makes the server look faster than it is.
--data SPEC         any raw GuideLLM data spec, passed through unchanged.


USAGE
-----
    # 1. one sweep per server (run again with the other server for a comparison)
    python throughput_interactivity.py collect \\
        --target http://localhost:8010 --model Qwen/Qwen3.8-27B \\
        --tokenizer Qwen/Qwen3.8-27B \\
        --dataset RedHatAI/speculator_benchmarks --subset HumanEval \\
        --max-tokens 1024 --streams 1,2,4,8,16,32,64,128 --repeats 1 \\
        --max-seconds 90 --warmup-seconds 30 \\
        --out-dir runs/baseline_humaneval --label baseline

    # 2. (or skip step 1) turn existing GuideLLM JSONs into a tidy CSV
    python throughput_interactivity.py parse runs/baseline_humaneval \\
        --label baseline --csv baseline_humaneval.csv

    # 3. sanity-check the sweep before you believe it
    python throughput_interactivity.py validate baseline_humaneval.csv

    # 4. plot one or more CSVs
    python throughput_interactivity.py plot \\
        --series baseline_humaneval.csv:'no speculator':'#7a8c3f' \\
        --series dspark_humaneval.csv:'dspark':'#b52513' \\
        --title 'Qwen3.8-27B, HumanEval' --subtitle '1x B200, vLLM ...' \\
        --out humaneval.png


READ THIS BEFORE QUOTING A NUMBER OFF THE CHART
-----------------------------------------------
`validate` checks all of these; each one silently produced a wrong chart at
least once.

1. Open loop: achieved rate must track offered rate.
2. Open loop: a plateau is usually a concurrency cap, not capacity.
3. A benchmark that ends on `requests_exhausted` ran out of dataset.
4. Steady state: tokens generated in the window per completed request should
   match the mean output length; a gap means the window was too short.
5. The synchronous point measures ~0.9995 concurrent, not 1.0.
6. Closed loop, fixed lengths: lockstep waves. `arrival_burstiness` above 2
   means the streams are synchronized; jitter the lengths or use real data.
7. Datasets: a small subset repeated many times is served partly from the
   prefix cache (`mean_cached_tokens`). Disable prefix caching for a clean number.
8. Closed loop past the server's batch cap: when N streams exceed max-num-seqs
   the extra requests wait inside the server, so TTFT jumps while ITL does not.
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
import urllib.error
import urllib.request
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # only `plot` needs it
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]

# Thresholds used by `validate`.
RATE_DRIFT_TOLERANCE = 0.05  # achieved/offered may differ from 1 by this much
RATE_KEPT_UP = 0.9  # a point "kept up" when achieved/offered is above this
RATE_SATURATED_FACTOR = 1.5  # ignore points whose achieved rate exceeds offered by more
PLATEAU_FRACTION = 0.97  # points within this fraction of the top concurrency share it
STEADY_STATE_TOLERANCE = 0.10  # tokens per completed request vs mean output length
REPEAT_SPREAD_LIMIT = 0.05  # max relative spread of throughput across repeats
LOCKSTEP_BURSTINESS = 2.0  # coefficient of variation of starts per second
LOCKSTEP_MIN_RPS = 1.0  # the burstiness check needs at least this many starts/s
CACHE_HIT_FRACTION = 0.05  # cached prompt tokens / prompt tokens
QUEUE_TTFT_MS = 1000.0  # a closed-loop point queues inside the server above this
QUEUE_TTFT_FACTOR = 10.0  # ... or above this multiple of the lowest TTFT
SYNC_CONCURRENCY = 1.0

# ---------------------------------------------------------------------------
# parse: GuideLLM JSON -> tidy rows
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model",
    "point",
    "offered_rps",
    "offered_concurrency",
    "repeat",
    "source_json",
    "strategy",
    "measured_duration_s",
    "successful_requests",
    "errored_requests",
    "incomplete_requests",
    "started_requests",
    "achieved_rps",
    "completed_rps",
    "aggregate_output_tps",
    "mean_active_concurrency",
    "interactivity_tps_user",
    "mean_itl_ms",
    "interactivity_itl_tps_user",
    "mean_output_tokens",
    "mean_prompt_tokens",
    "mean_cached_tokens",
    "mean_output_tokens_per_iteration",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_itl_ms",
    "arrival_burstiness",
    "acceptance_length",
    "num_drafts",
    "num_accepted_tokens",
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
    timings = (request.get("info") or {}).get("timings") or {}
    first = timings.get("first_token_iteration") or timings.get(
        "first_output_token_iteration"
    )
    last = timings.get("last_token_iteration")
    if first is None or last is None:  # older GuideLLM: approximate with TTFT
        ttft = request.get("time_to_first_token_ms") or 0.0
        first = request["request_start_time"] + ttft / 1000.0
        last = request["request_end_time"]
    return first, last


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


def _arrival_burstiness(requests: list, start: float, end: float) -> float:
    """Coefficient of variation of request starts per 1 s bin inside the window.

    About 0.1 for a constant-rate generator, 1/sqrt(mean) for Poisson arrivals,
    and above 2 when closed-loop streams with identical lengths finish and
    restart together (lockstep waves; see rule 6).
    """
    bins = int(end - start)
    if bins < 2:  # noqa: PLR2004
        return float("nan")
    counts = [0] * bins
    for r in requests:
        t = r["request_start_time"]
        if start <= t < end:
            counts[min(int(t - start), bins - 1)] += 1
    mean_count = statistics.fmean(counts)
    if mean_count <= 0:
        return float("nan")
    return statistics.pstdev(counts) / mean_count


def summarize_benchmark(benchmark: dict) -> dict:
    """Reduce one GuideLLM benchmark object to the metrics the chart needs."""
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
    started = sum(1 for r in every if start <= r["request_start_time"] < end)
    completed = sum(1 for r in successful if start <= r["request_end_time"] < end)

    aggregate_tps = window_tokens / window
    concurrency = in_flight / window
    strategy = benchmark["config"]["strategy"]
    state = benchmark.get("scheduler_state", {})
    constraints = state.get("end_processing_constraints", {})
    output_tokens = sum(r["output_tokens"] or 0 for r in successful)
    itls = [r.get("inter_token_latency_ms") for r in successful]
    itls = [v for v in itls if v]
    mean_itl = _mean(itls)
    ttfts = [r.get("time_to_first_token_ms") for r in successful]
    # Output tokens per streamed chunk: ~1 without speculative decoding, the
    # accepted length + 1 with it (vLLM streams every token a step produced at
    # once). A client-side cross-check for the server's acceptance counters.
    per_iteration = []
    for r in successful:
        timings = (r.get("info") or {}).get("timings") or {}
        iterations = timings.get("token_iterations")
        if iterations and (r.get("output_tokens") or 0) > 0:
            per_iteration.append(r["output_tokens"] / iterations)
    cached = [r.get("cached_tokens") for r in successful]
    streams = strategy.get("streams")
    if strategy.get("type_") == "synchronous":
        streams = 1

    return {
        "strategy": strategy.get("type_", ""),
        "offered_rps": strategy.get("rate"),
        "offered_concurrency": streams,
        "measured_duration_s": window,
        "successful_requests": len(successful),
        # list sizes, not scheduler_metrics.requests_made, which in GuideLLM 0.8
        # also counts requests that were queued but never sent
        "errored_requests": len(errored),
        "incomplete_requests": len(incomplete),
        "started_requests": started,
        "achieved_rps": started / window,
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
        "mean_output_tokens": (
            output_tokens / len(successful) if successful else float("nan")
        ),
        "mean_prompt_tokens": _mean([r.get("prompt_tokens") for r in successful]),
        "mean_cached_tokens": _mean(cached),
        "mean_output_tokens_per_iteration": _mean(per_iteration),
        "median_ttft_ms": _median(ttfts),
        "p99_ttft_ms": _percentile(ttfts, 0.99),
        "median_itl_ms": _median([r.get("inter_token_latency_ms") for r in successful]),
        "arrival_burstiness": _arrival_burstiness(every, start, end),
        "stop_reason": ",".join(constraints.keys()),
    }


def read_acceptance(json_path: Path) -> dict:
    """Acceptance metrics `collect` stored next to a run, if any."""
    sidecar = json_path.with_name(json_path.stem + METRICS_SUFFIX)
    empty = {
        "acceptance_length": float("nan"),
        "num_drafts": float("nan"),
        "num_accepted_tokens": float("nan"),
    }
    if not sidecar.is_file():
        return empty
    try:
        delta = json.loads(sidecar.read_text()).get("delta") or {}
    except json.JSONDecodeError:
        return empty
    return {
        "acceptance_length": delta.get("acceptance_length", float("nan")),
        "num_drafts": delta.get("num_drafts", float("nan")),
        "num_accepted_tokens": delta.get("num_accepted_tokens", float("nan")),
    }


def _point_and_repeat(stem: str) -> tuple[str, int]:
    """`<point>_r<N>` -> (point, N); anything else -> (stem, 1)."""
    if "_r" in stem:
        head, _, tail = stem.rpartition("_r")
        if tail.isdigit():
            return head, int(tail)
    return stem, 1


def parse_dir(json_dir: Path, label: str) -> list[dict]:
    """Summarize every GuideLLM benchmark in every `*.json` under `json_dir`.

    The point name and repeat come from file names of the form `<point>_r<N>.json`
    (what `collect` writes). A synchronous benchmark is renamed `sync` and a
    concurrent one `conc<N>` whatever the file is called.
    """
    rows = []
    for path in sorted(json_dir.rglob("*.json")):
        if path.name.endswith(METRICS_SUFFIX):
            continue
        point, repeat = _point_and_repeat(path.stem)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        benchmarks = payload.get("benchmarks") or []
        for index, benchmark in enumerate(benchmarks):
            try:
                summary = summarize_benchmark(benchmark)
            except (KeyError, ValueError) as exc:
                print(f"  skip {path.name}[{index}]: {exc}", file=sys.stderr)
                continue
            name = point if len(benchmarks) == 1 else f"{point}#{index}"
            if len(benchmarks) == 1:
                if summary["strategy"] == "synchronous":
                    name = "sync"
                elif (
                    summary["strategy"] == "concurrent"
                    and summary["offered_concurrency"]
                ):
                    name = f"conc{summary['offered_concurrency']:g}"
            summary.update(read_acceptance(path))
            summary.update(
                model=label, point=name, repeat=repeat, source_json=path.name
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
# server metrics: speculative decoding acceptance from vLLM's /metrics
# ---------------------------------------------------------------------------

SPEC_PREFIX = "vllm:spec_decode_"
_RE_METRIC = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)$")
_RE_POSITION = re.compile(r'position="(\d+)"')


def fetch_text(url: str, timeout: float = 30.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def spec_decode_counters(text: str) -> dict:
    """Sum vLLM's speculative-decoding counters over engines.

    Reads `vllm:spec_decode_num_drafts`, `..._num_draft_tokens`,
    `..._num_accepted_tokens` and `..._num_accepted_tokens_per_pos` (with or
    without the `_total` suffix) and returns the sums plus the per-position list.
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
        name = match.group(1).replace("_total", "")
        if not name.startswith(SPEC_PREFIX):
            continue
        labels, value = match.group(2) or "", float(match.group(3))
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
    }


def acceptance_delta(before: dict, after: dict) -> dict:
    """Acceptance over one run from two snapshots of the cumulative counters."""
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
    }


# ---------------------------------------------------------------------------
# collect: drive GuideLLM
# ---------------------------------------------------------------------------

DEFAULT_COLUMN_MAPPER = (
    "kind=generative_column_mapper,column_mappings.text_column={column}"
)
HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/main/{subset}.jsonl"
MIN_DATASET_REQUESTS = 5000
REQUESTS_PER_STREAM = 20


def _dataset_rows(dataset: str, subset: str | None) -> list:
    """Rows of `<subset>.jsonl` from a local file or directory, or a HF dataset id."""
    local = Path(dataset)
    candidates = [local / f"{subset}.jsonl", local] if subset else [local]
    for candidate in candidates:
        if candidate.is_file():
            lines = candidate.read_text().splitlines()
            return [json.loads(line) for line in lines if line.strip()]
    if local.exists():
        raise FileNotFoundError(f"{dataset}: no {subset}.jsonl inside, and not a file")
    if not subset:
        raise ValueError("--subset is required with a Hugging Face dataset id")
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
        target = max(
            MIN_DATASET_REQUESTS, REQUESTS_PER_STREAM * max(args.streams or [0])
        )
        repeat = max(1, math.ceil(target / len(rows)))
    prompts = [{"prompt": r[args.prompt_column]} for r in rows] * repeat
    random.Random(0).shuffle(prompts)
    stem = Path(args.subset or args.dataset).stem
    path = out_dir / "_data" / f"{stem}_x{repeat}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or args.overwrite:
        with path.open("w") as handle:
            for row in prompts:
                handle.write(json.dumps(row) + "\n")
    print(f"dataset: {len(rows)} rows x {repeat} = {len(prompts)} prompts -> {path}")
    return path


def build_data_args(args: argparse.Namespace, out_dir: Path) -> list[str]:
    """The `--data` (and `--data-column-mapper`) arguments for GuideLLM."""
    modes = [
        bool(args.data),
        bool(args.prompt_tokens or args.output_tokens),
        bool(args.dataset),
    ]
    if sum(modes) != 1:
        raise SystemExit(
            "choose exactly one of --data, --prompt-tokens/--output-tokens, "
            "or --dataset"
        )
    mapper = args.data_column_mapper
    if args.data:
        spec = args.data
    elif args.dataset:
        path = materialize_dataset(args, out_dir)
        spec = f"kind=json_file,path={path},load_kwargs.split=train"
        mapper = mapper or DEFAULT_COLUMN_MAPPER.format(column="prompt")
    else:
        if not (args.prompt_tokens and args.output_tokens):
            raise SystemExit("--prompt-tokens and --output-tokens go together")
        p, o, ratio = args.prompt_tokens, args.output_tokens, args.range_ratio
        spec = f"kind=synthetic_text,prompt_tokens={p},output_tokens={o}"
        if ratio < 1.0:  # uniform in [ratio*L, L], the InferenceX convention (0.8)
            spec += (
                f",prompt_tokens_min={int(p * ratio)},prompt_tokens_max={p}"
                f",output_tokens_min={int(o * ratio)},output_tokens_max={o}"
            )
    text = ["--data", spec]
    if mapper:
        text += ["--data-column-mapper", mapper]
    return text


def build_backend(args: argparse.Namespace) -> str:
    """GuideLLM backend spec as JSON, so nested `extras` survive."""
    default_format = "/v1/chat/completions" if args.dataset else "/v1/completions"
    backend: dict[str, Any] = {
        "kind": "openai_http",
        "target": args.target,
        "model": args.model,
        "request_format": args.request_format or default_format,
    }
    if args.max_tokens:
        backend["max_tokens"] = args.max_tokens
    if args.ignore_eos:
        backend["extras"] = {"body": {"ignore_eos": True}}
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
    """`<out_dir>/bench_command.txt`: enough to rerun and to cite the run."""
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
        text = fetch_text(args.target.rstrip("/") + path, timeout=10)
        lines.append(f"{name}: {' '.join(text.split()) if text else 'unavailable'}")
    (out_dir / "bench_command.txt").write_text("\n".join(lines) + "\n")


def _points(args: argparse.Namespace) -> list[tuple[str, str]]:
    warmup = f",warmup={args.warmup_seconds:g}" if args.warmup_seconds else ""
    points: list[tuple[str, str]] = []
    synchronous = args.synchronous
    if synchronous is None:  # closed-loop sweeps include streams=1, the same point
        synchronous = not args.streams
    if synchronous:
        points.append(("sync", f"kind=synchronous{warmup}"))
    for streams in args.streams or []:
        points.append(
            (f"conc{streams:g}", f"kind=concurrent,streams={streams:g}{warmup}")
        )
    for rate in args.rates or []:
        points.append((f"rate{rate:g}", f"kind=constant,rate={rate:g}{warmup}"))
    if not points:
        raise SystemExit(
            "nothing to run: give --streams and/or --rates (or --synchronous)"
        )
    return points


def collect(args: argparse.Namespace) -> int:  # noqa: C901
    """Run one GuideLLM invocation per (point, repeat).

    Points are `sync`, `conc<N>` for each --streams value (closed loop) and
    `rate<r>` for each --rates value (open loop). Existing JSONs are skipped unless
    --overwrite, so a sweep can be resumed or extended into the same directory.
    Around every run the server's /metrics counters are snapshotted, and the
    acceptance over the run is stored in `<point>_r<N>.metrics.json`.
    """
    guidellm = shutil.which(args.guidellm_bin)
    if guidellm is None and not args.dry_run:
        raise SystemExit(
            f"{args.guidellm_bin!r} not found on PATH; pip install guidellm"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = _points(args)
    data_args = build_data_args(args, out_dir)
    backend = build_backend(args)
    total_seconds = args.max_seconds + (args.warmup_seconds or 0)
    metrics_url = (
        None
        if args.no_metrics
        else (args.metrics_url or args.target.rstrip("/") + "/metrics")
    )
    if not args.dry_run:
        write_provenance(out_dir, args)

    failures = 0
    for point, profile in points:
        for repeat in range(1, args.repeats + 1):
            target_json = out_dir / f"{point}_r{repeat}.json"
            if target_json.exists() and not args.overwrite:
                print(f"exists, skipping: {target_json.name}")
                continue
            command = [
                guidellm or args.guidellm_bin,
                "run",
                "--backend",
                backend,
                "--tokenizer",
                f"kind=hf_auto,model={args.tokenizer or args.model}",
                *data_args,
                "--profile",
                profile,
                "--constraint",
                f"kind=max_duration,seconds={total_seconds:g}",
                "--metrics",
                "kind=generative,sample_size=0",
                "--output",
                f"kind=json,path={target_json}",
                "--disable-console-interactive",
                *args.guidellm_arg,
            ]
            print(f"\n=== {point} repeat {repeat}\n{shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            before = fetch_text(metrics_url) if metrics_url else None
            result = subprocess.run(command, check=False)  # noqa: S603
            if result.returncode != 0:
                failures += 1
                print(
                    f"!! exit {result.returncode} for {point} r{repeat}",
                    file=sys.stderr,
                )
                if not args.keep_going:
                    return result.returncode
                continue
            after = fetch_text(metrics_url) if metrics_url else None
            if before is not None and after is not None:
                snapshot_before = spec_decode_counters(before)
                snapshot_after = spec_decode_counters(after)
                sidecar = target_json.with_name(target_json.stem + METRICS_SUFFIX)
                sidecar.write_text(
                    json.dumps(
                        {
                            "metrics_url": metrics_url,
                            "before": snapshot_before,
                            "after": snapshot_after,
                            "delta": acceptance_delta(snapshot_before, snapshot_after),
                        },
                        indent=1,
                    )
                )
            elif metrics_url:
                print(f"(no /metrics at {metrics_url}; acceptance not recorded)")

    if args.dry_run:
        print("\n(dry run, nothing executed)")
        return 0

    rows = parse_dir(out_dir, args.label)
    if not rows:
        print("no benchmarks parsed", file=sys.stderr)
        return 1
    csv_path = Path(args.csv) if args.csv else out_dir / f"{args.label}.csv"
    write_csv(rows, csv_path)
    print(f"\nwrote {len(rows)} rows -> {csv_path}")
    validate_rows(rows)
    return 1 if failures and not args.keep_going else 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "aggregate_output_tps": [
        "aggregate_output_tps",
        "output_throughput_tok_s",
        "output_tps",
    ],
    "interactivity_itl_tps_user": ["interactivity_itl_tps_user", "interactivity_itl"],
    "interactivity_tps_user": [
        "interactivity_tps_user",
        "interactivity_tok_s_per_active_request",
        "interactivity",
    ],
    "mean_active_concurrency": [
        "mean_active_concurrency",
        "mean_active_successful_concurrency",
        "concurrency",
    ],
    "offered_concurrency": ["offered_concurrency", "streams"],
    "completed_rps": ["completed_rps", "achieved_rps"],
}


def _pick(row: dict, canonical: str) -> Any:
    for name in COLUMN_ALIASES.get(canonical, [canonical]):
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [r for r in csv.DictReader(handle) if any(v for v in r.values())]


def _point_name(row: dict) -> str:
    """Stable name for one load level: `sync`, `conc<N>` or `rate<r>`."""
    raw = str(_pick(row, "point") or "")
    strategy = str(_pick(row, "strategy") or "")
    offered = _as_float(_pick(row, "offered_rps"))
    streams = _as_float(_pick(row, "offered_concurrency"))
    if "sync" in raw.lower() or strategy == "synchronous":
        return "sync"
    if raw.startswith("conc"):
        return raw
    if strategy == "concurrent" and streams:
        return f"conc{streams:g}"
    if not offered:
        return "sync"
    return f"rate{offered:g}"


def _group_mean(group: list[dict], field: str) -> float | None:
    values = [_as_float(_pick(r, field)) for r in group]
    values = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(values) if values else None


AGGREGATE_FIELDS = {
    "x_itl": "interactivity_itl_tps_user",
    "x_little": "interactivity_tps_user",
    "y": "aggregate_output_tps",
    "rps": "achieved_rps",
    "completed_rps": "completed_rps",
    "offered": "offered_rps",
    "streams": "offered_concurrency",
    "conc": "mean_active_concurrency",
    "itl": "median_itl_ms",
    "itl_mean": "mean_itl_ms",
    "ttft": "median_ttft_ms",
    "ttft_p99": "p99_ttft_ms",
    "tokens": "mean_output_tokens",
    "prompt_tokens": "mean_prompt_tokens",
    "cached": "mean_cached_tokens",
    "per_iteration": "mean_output_tokens_per_iteration",
    "burst": "arrival_burstiness",
    "acceptance": "acceptance_length",
    "incomplete": "incomplete_requests",
    "successful": "successful_requests",
}


def aggregate_points(rows: list[dict]) -> list[dict]:
    """Average repeats of the same point; carry the per-repeat spread."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_point_name(row)].append(row)

    points = []
    for name, group in groups.items():
        throughputs = [_as_float(_pick(r, "aggregate_output_tps")) for r in group]
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
            x=point["x_itl"] if point["x_itl"] is not None else point["x_little"],
            stop_reason=_pick(group[0], "stop_reason") or "",
            synchronous=name.startswith("sync"),
            closed_loop=name.startswith("conc"),
        )
        points.append(point)
    points.sort(key=lambda p: (p["conc"] is None, p["conc"] or 0.0))
    return points


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def _print_table(points: list[dict]) -> None:
    header = (
        "point",
        "offered",
        "achieved",
        "ratio",
        "tok/s",
        "conc",
        "1/itl",
        "little",
        "itl_ms",
        "ttft_ms",
        "accept",
        "burst",
        "spread",
    )
    layout = (
        "{:>9} {:>8} {:>8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6} {:>6}"
    )
    print("\n" + layout.format(*header))
    for p in points:
        open_loop = not p["synchronous"] and not p["closed_loop"]
        ratio = (
            p["rps"] / p["offered"] if open_loop and p["rps"] and p["offered"] else None
        )
        if p["synchronous"]:
            offered = "sync"
        elif p["closed_loop"]:
            offered = f"N={p['streams']:.0f}" if p["streams"] else "N=?"
        else:
            offered = _fmt(p["offered"], ".2f")
        print(
            layout.format(
                p["point"],
                offered,
                _fmt(p["rps"], ".2f"),
                _fmt(ratio, ".3f"),
                _fmt(p["y"], ".0f"),
                _fmt(p["conc"], ".1f"),
                _fmt(p["x_itl"], ".1f"),
                _fmt(p["x_little"], ".1f"),
                _fmt(p["itl_mean"], ".2f"),
                _fmt(p["ttft"], ".0f"),
                _fmt(p["acceptance"], ".2f"),
                _fmt(p["burst"], ".2f"),
                f"{100 * p['spread']:.1f}%",
            )
        )


def validate_rows(rows: list[dict]) -> list[str]:  # noqa: C901
    points = aggregate_points(rows)
    warnings: list[str] = []
    open_loop = [p for p in points if not p["synchronous"] and not p["closed_loop"]]
    closed_loop = [p for p in points if p["closed_loop"]]
    _print_table(points)

    # 1. open loop: offered vs achieved
    ratios = [
        (p["point"], p["rps"] / p["offered"])
        for p in open_loop
        if p["rps"]
        and p["offered"]
        and p["rps"] <= p["offered"] * RATE_SATURATED_FACTOR
    ]
    kept_up = [r for _, r in ratios if r > RATE_KEPT_UP]
    if kept_up:
        drift = statistics.fmean(kept_up)
        if abs(drift - 1.0) > RATE_DRIFT_TOLERANCE:
            warnings.append(
                f"offered rate looks mis-scaled: achieved/offered averages {drift:.3f} "
                "on points that kept up. The generator is not sending its nominal "
                "rate, so per-rate comparisons with another sweep are invalid."
            )

    # 2. open loop: concurrency cap / plateau (closed loop pins concurrency to N)
    if len(open_loop) >= 2:  # noqa: PLR2004
        top = max(p["conc"] or 0 for p in open_loop)
        at_cap = [
            p for p in open_loop if p["conc"] and p["conc"] > PLATEAU_FRACTION * top
        ]
        if len(at_cap) >= 2:  # noqa: PLR2004
            names = ", ".join(p["point"] for p in at_cap)
            warnings.append(
                f"{len(at_cap)} points sit at the same concurrency (~{top:.0f}): "
                f"{names}. "
                "That plateau is a concurrency cap, so peak throughput here is "
                "cap/latency, not capacity. Raise the client's max concurrency and "
                "the server's max-num-seqs and re-run the top points."
            )

    # 3. dataset exhaustion
    exhausted = [p["point"] for p in points if "requests_exhausted" in p["stop_reason"]]
    if exhausted:
        shown = ", ".join(exhausted[:4]) + ("..." if len(exhausted) > 4 else "")  # noqa: PLR2004
        warnings.append(
            f"{len(exhausted)} points ended on requests_exhausted ({shown}): the sweep "
            "ran out of dataset, so its concurrency is bounded by dataset size, not "
            "the server. With --dataset, raise --dataset-repeat."
        )

    # 4. steady state: tokens in the window per completed request vs output length
    for p in points:
        if p["y"] and p["completed_rps"] and p["tokens"]:
            accounted = p["y"] / p["completed_rps"]
            if abs(accounted - p["tokens"]) > STEADY_STATE_TOLERANCE * p["tokens"]:
                warnings.append(
                    f"{p['point']}: the window holds {accounted:.0f} generated tokens "
                    f"per completed request but requests average {p['tokens']:.0f}; "
                    "the run did not reach steady state within the window. Lengthen "
                    "--max-seconds and --warmup-seconds."
                )
                break

    # 5. repeat noise
    noisy = [
        p for p in points if p["repeats"] > 1 and p["spread"] > REPEAT_SPREAD_LIMIT
    ]
    if noisy:
        worst = max(noisy, key=lambda p: p["spread"])
        warnings.append(
            f"{len(noisy)} points vary more than {100 * REPEAT_SPREAD_LIMIT:.0f}% "
            "across "
            f"repeats (worst {worst['spread'] * 100:.1f}% at {worst['point']}): treat "
            "differences smaller than that as noise."
        )

    # 6. lockstep waves: requests start in bursts instead of spread over the window
    waves = [
        p
        for p in points
        if p["burst"]
        and p["burst"] > LOCKSTEP_BURSTINESS
        and (p["rps"] or 0) >= LOCKSTEP_MIN_RPS
    ]
    if waves:
        names = ", ".join(f"{p['point']} (cv {p['burst']:.1f})" for p in waves[:5])
        warnings.append(
            f"{len(waves)} points start their requests in synchronized waves: {names}. "
            "Streams with identical lengths finish together, prefill in one burst and "
            "then decode with no prefill in the batch, which overstates throughput and "
            "ITL. Jitter the lengths (--range-ratio 0.8) or use a dataset."
        )

    # 7. prefix cache hits on repeated prompts
    cached = [
        p
        for p in points
        if p["cached"]
        and p["prompt_tokens"]
        and p["cached"] / p["prompt_tokens"] > CACHE_HIT_FRACTION
    ]
    if cached:
        worst = max(cached, key=lambda p: p["cached"] / p["prompt_tokens"])
        share = 100 * worst["cached"] / worst["prompt_tokens"]
        warnings.append(
            f"{len(cached)} points were served partly from the prefix cache (up to "
            f"{share:.0f}% of prompt tokens at {worst['point']}). Prefill work is "
            "smaller than the prompt length suggests; disable prefix caching on the "
            "server or report these as cached-workload numbers."
        )

    # 8. closed loop past the server's batch cap: queueing shows up as TTFT
    ttfts = [p["ttft"] for p in points if p["ttft"]]
    if closed_loop and ttfts:
        floor = max(QUEUE_TTFT_MS, QUEUE_TTFT_FACTOR * min(ttfts))
        queued = [p for p in closed_loop if p["ttft"] and p["ttft"] > floor]
        if queued:
            names = ", ".join(
                f"{p['point']} ({p['ttft'] / 1000:.1f} s)" for p in queued[:4]
            )
            warnings.append(
                f"{len(queued)} closed-loop points have a median TTFT above "
                f"{floor / 1000:.1f} s: {names}. More streams than the server "
                "admits at "
                "once (max-num-seqs) wait inside the server, so these points measure "
                "queueing plus decode. Raise max-num-seqs or stop the sweep below "
                "this N."
            )

    print()
    for warning in warnings:
        print(f"WARNING: {warning}\n")
    if not warnings:
        print("no warnings\n")
    return warnings


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

BG, GRID, AXIS = "#f2f2f2", "#cfdae2", "#888888"
INK, MUTED = "#181818", "#666666"
PALETTE = ["#b52513", "#2c6e9b", "#7a8c3f", "#b8b1a8", "#8a5a9e"]
CANVAS = (2160, 1216)
PLOT = (175, 230, 2010, 1030)  # left, top, right, bottom
FONT_CANDIDATES = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}
# label slots tried around each marker, nearest first
LABEL_OFFSETS = [
    (24, -30),
    (24, 0),
    (24, 30),
    (-24, -30),
    (-24, 0),
    (-24, 30),
    (24, -60),
    (-24, -60),
    (24, 60),
    (-24, 60),
    (24, -90),
    (-24, -90),
    (60, -30),
    (-60, -30),
    (60, 30),
    (-60, 30),
]
LABEL_HEIGHT = 24
AXIS_TICKS = 7
THOUSAND = 1000
RATE_LABEL_DECIMALS = 10


def _font(kind: str, size: int) -> Any:
    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:  # Pillow >= 10.1 ships a scalable default font
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _nice_axis_max(value: float, ticks: int = AXIS_TICKS) -> tuple[float, float]:
    """Round up to a readable tick step, leaving headroom past the last tick."""
    if value <= 0:
        return 1.0, 1.0
    raw = value / ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= raw:
            break
    tick_max = math.ceil(value / step) * step
    if tick_max < value * 1.001:
        tick_max += step
    return (tick_max - step if tick_max - step >= value else tick_max), step


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    if value >= THOUSAND:
        text = f"{value / THOUSAND:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    return f"{value:.10g}"


def _segments(series_coords: list) -> list:
    out = []
    for coords in series_coords:
        out.extend(zip(coords, coords[1:], strict=False))
    return out


def _box_hits_segment(box: tuple, p0: tuple, p1: tuple) -> bool:
    x0, y0, x1, y1 = box
    (ax, ay), (bx, by) = p0, p1
    if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
        return False
    steps = max(2, int(max(abs(bx - ax), abs(by - ay)) / 6))
    for index in range(steps + 1):
        t = index / steps
        px, py = ax + (bx - ax) * t, ay + (by - ay) * t
        if x0 <= px <= x1 and y0 <= py <= y1:
            return True
    return False


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _point_label(point: dict, label_format: str) -> str:
    rps, conc = point["rps"] or 0.0, point["conc"] or 0.0
    streams = point["streams"] or 0.0
    if point["synchronous"]:
        return "single stream"
    if label_format != "auto":
        return label_format.format(rps=rps, conc=conc, streams=streams)
    if point["closed_loop"]:
        return f"{streams:.0f} concurrent"
    # two decimals under 1 req/s and one under 10, so 0.25 does not read as 0.2
    if rps < 1:
        return f"{rps:.2f}".rstrip("0").rstrip(".") + " req/s"
    return f"{rps:.1f} req/s" if rps < RATE_LABEL_DECIMALS else f"{rps:.0f} req/s"


def _load_series(args: argparse.Namespace) -> list[dict]:
    series = []
    for spec in args.series:
        path_text, name, color = (spec.split(":") + ["", ""])[:3]
        path = Path(path_text)
        name = name or path.stem
        color = color or PALETTE[len(series) % len(PALETTE)]
        points = aggregate_points(load_rows(path))
        for p in points:
            p["x"] = p["x_itl"] if args.x == "itl" else p["x_little"]
        missing = [p["point"] for p in points if not p["x"] and p["y"]]
        if missing and args.x == "itl":
            print(
                f"{path}: no mean_itl_ms for {', '.join(missing[:4])}; re-parse the "
                "JSONs or use --x little",
                file=sys.stderr,
            )
        points = [p for p in points if p["x"] and p["y"]]
        if args.max_concurrency is not None:
            points = [p for p in points if (p["conc"] or 0) <= args.max_concurrency]
        if args.min_concurrency is not None:
            points = [p for p in points if (p["conc"] or 0) >= args.min_concurrency]
        drop = set(args.drop or [])
        points = [p for p in points if p["point"] not in drop]
        if not points:
            raise SystemExit(f"no points left for {path}")
        # draw the line in load order (concurrency), not by throughput: past
        # saturation throughput can fall while concurrency keeps growing
        points.sort(key=lambda p: p["conc"] or 0.0)
        series.append({"name": name, "color": color, "points": points})
    return series


def plot(args: argparse.Namespace) -> int:  # noqa: C901
    if Image is None:
        raise SystemExit("plot needs Pillow: pip install pillow")
    series = _load_series(args)
    every = [p for s in series for p in s["points"]]
    x_tick_max, x_step = _nice_axis_max(max(p["x"] for p in every))
    y_tick_max, y_step = _nice_axis_max(max(p["y"] for p in every))
    x_max = x_tick_max + x_step * 0.4
    y_max = y_tick_max + y_step * 0.4

    left, top, right, bottom = PLOT
    image = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(image)
    f_title, f_sub = _font("bold", 48), _font("regular", 26)
    f_axis, f_tick = _font("regular", 25), _font("regular", 22)
    f_label, f_legend = _font("bold", 19), _font("regular", 24)

    def tw(text: str, font: Any) -> float:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0]

    def sx(value: float) -> float:
        return left + value / x_max * (right - left)

    def sy(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    xlabel = args.xlabel or (
        "Interactivity (tok/s/user, 1 / mean inter-token latency)"
        if args.x == "itl"
        else "Interactivity (tok/s/user, throughput / requests in flight)"
    )
    draw.text((112, 48), args.title, font=f_title, fill=INK)
    if args.subtitle:
        draw.text((112, 112), args.subtitle, font=f_sub, fill=MUTED)

    for index in range(int(round(x_tick_max / x_step)) + 1):
        value = index * x_step
        x = sx(value)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        text = _format_tick(value)
        draw.text(
            (x - tw(text, f_tick) / 2, bottom + 15), text, font=f_tick, fill=MUTED
        )
    for index in range(int(round(y_tick_max / y_step)) + 1):
        value = index * y_step
        y = sy(value)
        draw.line((left, y, right, y), fill=GRID, width=2)
        text = _format_tick(value)
        draw.text((left - 18 - tw(text, f_tick), y - 12), text, font=f_tick, fill=MUTED)
    draw.line((left, top, left, bottom), fill=AXIS, width=2)
    draw.line((left, bottom, right, bottom), fill=AXIS, width=2)

    draw.text(
        ((CANVAS[0] - tw(xlabel, f_axis)) / 2, 1090), xlabel, font=f_axis, fill=INK
    )
    box = draw.textbbox((0, 0), args.ylabel, font=f_axis)
    layer = Image.new("RGBA", (box[2] + 8, box[3] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((4, 0), args.ylabel, font=f_axis, fill=INK)
    layer = layer.rotate(90, expand=True)
    image.paste(layer, (48, int((top + bottom - layer.height) / 2)), layer)
    draw = ImageDraw.Draw(image)

    for entry in series:
        entry["coords"] = [(sx(p["x"]), sy(p["y"])) for p in entry["points"]]

    legend_rows = [(e["name"], e["color"]) for e in series]
    legend_box = None
    if len(series) > 1 or args.force_legend:
        width = 78 + max(tw(t, f_legend) for t, _ in legend_rows) + 30
        height = 22 + 40 * len(legend_rows)
        lx, ly = right - width - 24, top + 20
        legend_box = (lx, ly, lx + width, ly + height)

    # Automatic label placement: try slots around each marker and keep the first
    # that hits no curve, marker, placed label, legend or plot edge.
    obstacles = _segments([e["coords"] for e in series])
    markers = [c for e in series for c in e["coords"]]
    placed: list[tuple] = [legend_box] if legend_box else []
    for index, entry in enumerate(series):
        if args.label_first_only and index > 0:
            continue
        for point, (mx, my) in zip(entry["points"], entry["coords"], strict=True):
            text = _point_label(point, args.label_format)
            width = tw(text, f_label)
            best, best_cost = None, None
            for slot, (dx, dy) in enumerate(LABEL_OFFSETS):
                tx = mx + dx if dx > 0 else mx + dx - width
                ty = my + dy - LABEL_HEIGHT / 2
                box = (tx - 3, ty - 3, tx + width + 3, ty + LABEL_HEIGHT + 3)
                if (
                    box[0] < left + 2
                    or box[2] > right - 2
                    or box[1] < top + 2
                    or box[3] > bottom - 2
                ):
                    continue
                cost = 8 * sum(
                    1 for p0, p1 in obstacles if _box_hits_segment(box, p0, p1)
                )
                cost += 8 * sum(
                    1
                    for cx, cy in markers
                    if box[0] - 8 < cx < box[2] + 8 and box[1] - 8 < cy < box[3] + 8
                )
                cost += 8 * sum(1 for other in placed if _boxes_overlap(box, other))
                cost += slot * 0.1  # prefer earlier (closer) slots
                if best_cost is None or cost < best_cost:
                    best, best_cost = (tx, ty, box), cost
                if cost < 0.9:  # noqa: PLR2004
                    break
            if best is None:
                continue
            tx, ty, box = best
            placed.append(box)
            draw.text((tx, ty), text, font=f_label, fill=INK)

    for entry in series:
        coords, color = entry["coords"], entry["color"]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=6, joint="curve")
        for x, y in coords:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=BG, width=2)

    if legend_box:
        lx, ly, _, _ = legend_box
        draw.rounded_rectangle(
            legend_box, radius=5, fill="#ffffff", outline="#d2d2d2", width=2
        )
        for index, (text, color) in enumerate(legend_rows):
            cy = ly + 32 + index * 40
            draw.line((lx + 30, cy, lx + 62, cy), fill=color, width=11)
            draw.text((lx + 78, cy - 16), text, font=f_legend, fill=INK)

    if args.note:
        draw.text((112, 1145), args.note, font=_font("regular", 21), fill=MUTED)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, optimize=True)
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


def _comma_list(cast: type) -> Any:
    return lambda text: [cast(x) for x in text.split(",")]


def _add_collect_parser(sub: Any) -> None:
    c = sub.add_parser("collect", help="run a GuideLLM sweep, then parse + validate")
    c.add_argument("--target", required=True, help="e.g. http://127.0.0.1:8000")
    c.add_argument(
        "--model", required=True, help="model name as served (`model` field)"
    )
    c.add_argument(
        "--tokenizer", default="", help="tokenizer path or HF id (default: --model)"
    )
    c.add_argument(
        "--request-format",
        default=None,
        help="/v1/completions (default), or /v1/chat/completions (default with "
        "--dataset)",
    )
    load = c.add_argument_group("load (give --streams, --rates, or both)")
    load.add_argument(
        "--streams",
        type=_comma_list(int),
        default=None,
        help="closed loop, InferenceX style: requests kept in flight, e.g. 1,2,4,8,16",
    )
    load.add_argument(
        "--rates",
        type=_comma_list(float),
        default=None,
        help="open loop: constant arrival rates in req/s, e.g. 0.5,1,2,4",
    )
    load.add_argument(
        "--synchronous",
        action="store_true",
        default=None,
        help="also run a single-stream point (default: only without --streams)",
    )
    load.add_argument("--no-synchronous", dest="synchronous", action="store_false")
    load.add_argument("--repeats", type=int, default=3)
    load.add_argument(
        "--max-seconds", type=float, default=100.0, help="window per point"
    )
    load.add_argument("--warmup-seconds", type=float, default=30.0, help="excluded")
    data = c.add_argument_group(
        "data (choose one: a dataset, random text, or a raw spec)"
    )
    data.add_argument(
        "--dataset",
        default=None,
        help="HF dataset id (e.g. RedHatAI/speculator_benchmarks) or a local dir/jsonl",
    )
    data.add_argument(
        "--subset", default=None, help="file name without .jsonl, e.g. HumanEval"
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
        "--max-tokens", type=int, default=None, help="max_tokens per request"
    )
    data.add_argument(
        "--ignore-eos", action="store_true", help="force max_tokens on every request"
    )
    data.add_argument(
        "--prompt-tokens", type=int, default=None, help="random text: prompt length"
    )
    data.add_argument(
        "--output-tokens", type=int, default=None, help="random text: output length"
    )
    data.add_argument(
        "--range-ratio",
        type=float,
        default=0.8,
        help="random text: lengths uniform in [ratio*L, L]; 0.8 as InferenceX, "
        "1 = fixed",
    )
    data.add_argument(
        "--data", default=None, help="raw GuideLLM data spec, passed through"
    )
    data.add_argument(
        "--data-column-mapper",
        default=None,
        help="raw GuideLLM column mapper (default with --dataset: text_column=prompt)",
    )
    c.add_argument("--out-dir", required=True)
    c.add_argument("--label", required=True, help="series name recorded in the CSV")
    c.add_argument("--csv", default="", help="default: <out-dir>/<label>.csv")
    c.add_argument("--metrics-url", default=None, help="default: <target>/metrics")
    c.add_argument("--no-metrics", action="store_true", help="do not record acceptance")
    c.add_argument("--guidellm-bin", default="guidellm", help="GuideLLM executable")
    c.add_argument(
        "--guidellm-arg",
        action="append",
        default=[],
        help="extra argument appended to every GuideLLM command (repeatable)",
    )
    c.add_argument(
        "--dry-run", action="store_true", help="print commands only; do this first"
    )
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--keep-going", action="store_true")
    c.set_defaults(func=collect)


def _add_plot_parser(sub: Any) -> None:
    g = sub.add_parser("plot", help="render the chart")
    g.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="CSV[:NAME[:#COLOR]]",
        help="repeatable; first series is drawn on top",
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
    g.add_argument("--xlabel", default="", help="default depends on --x")
    g.add_argument("--ylabel", default="Output Token Throughput (tok/s)")
    g.add_argument("--note", default="", help="one grey line under the x-axis label")
    g.add_argument(
        "--label-format",
        default="auto",
        help="'auto', or a format string over {rps}, {conc} and {streams}",
    )
    g.add_argument(
        "--max-concurrency", type=float, default=None, help="drop points above"
    )
    g.add_argument(
        "--min-concurrency", type=float, default=None, help="drop points below"
    )
    g.add_argument("--drop", action="append", help="drop a point by name, repeatable")
    g.add_argument("--force-legend", action="store_true")
    g.add_argument(
        "--label-first-only",
        action="store_true",
        help="label only the first series' points (for curves that overlap)",
    )
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
