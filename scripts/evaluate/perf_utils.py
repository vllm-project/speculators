"""Shared utilities for evaluation and visualization scripts."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

try:
    from scipy.interpolate import UnivariateSpline
except ImportError:
    UnivariateSpline = None  # type: ignore[assignment,misc]

logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# Metric definitions (for plotting)
# ---------------------------------------------------------------------------

METRICS: dict[str, dict[str, str | bool]] = {
    "latency": {
        "csv_col": "latency_median_s",
        "json_key": "request_latency",
        "stat": "median",
        "label": "Median Latency (s)",
        "increasing": True,
    },
    "itl": {
        "csv_col": "itl_median_ms",
        "json_key": "inter_token_latency_ms",
        "stat": "median",
        "label": "Median ITL (ms)",
        "increasing": True,
    },
    "ttft": {
        "csv_col": "ttft_median_ms",
        "json_key": "time_to_first_token_ms",
        "stat": "median",
        "label": "Median TTFT (ms)",
        "increasing": True,
    },
    "output_tps": {
        "csv_col": "output_tps_median",
        "json_key": "output_tokens_per_second",
        "stat": "median",
        "label": "Output Tokens/s",
        "increasing": False,
    },
}

PRETTY_SUBSET_NAMES: dict[str, str] = {
    "HumanEval": "HumanEval",
    "math_reasoning": "Math Reasoning",
    "qa": "QA",
    "question": "Question",
    "rag": "RAG",
    "summarization": "Summarization",
    "tool_call": "Tool Call",
    "translation": "Translation",
    "writing": "Writing",
}

METRICS_TO_EXTRACT = [
    ("requests_per_second", "rps_median"),
    ("request_latency", "latency_median_s"),
    ("inter_token_latency_ms", "itl_median_ms"),
    ("time_to_first_token_ms", "ttft_median_ms"),
    ("output_tokens_per_second", "output_tps_median"),
]

SKIP_STRATEGIES = {"throughput"}

BASE_CSV_COLUMNS = [
    "subset",
    "strategy",
    "target_rate",
    "rps_median",
    "latency_median_s",
    "itl_median_ms",
    "ttft_median_ms",
    "output_tps_median",
    "total_output_tokens",
]

# ---------------------------------------------------------------------------
# Prometheus / spec-decode metric types
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    name: str


@dataclass
class Counter(Metric):
    value: float


@dataclass
class Vector(Metric):
    values: list[float]


# ---------------------------------------------------------------------------
# Data loading (CSV + JSON)
# ---------------------------------------------------------------------------


def _load_csv(
    filepath: Path,
    metric_name: str,
) -> dict[str, list[tuple[float, float]]]:
    cfg = METRICS[metric_name]
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with filepath.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("strategy") != "constant":
                continue
            try:
                subset = re.sub(r"^run_", "", row.get("subset", "unknown"))
                result[subset].append(
                    (float(row["rps_median"]), float(row[cfg["csv_col"]]))
                )
            except (ValueError, KeyError):
                continue
    return dict(result)


def _load_json(
    filepath: Path,
    metric_name: str,
) -> dict[str, list[tuple[float, float]]]:
    cfg = METRICS[metric_name]
    with filepath.open() as f:
        data = json.load(f)

    subset = Path(data["args"]["data_args"][0]["data_files"]).stem
    points: list[tuple[float, float]] = []
    for bench in data.get("benchmarks", []):
        if bench.get("config", {}).get("strategy", {}).get("type_") != "constant":
            continue
        try:
            rps = bench["metrics"]["requests_per_second"]["successful"]["mean"]
            y = bench["metrics"][cfg["json_key"]]["successful"][cfg["stat"]]
            points.append((rps, y))
        except (KeyError, TypeError):
            continue

    points.sort(key=lambda p: p[0])
    return {subset: points} if points else {}


def load_data(
    filepath: Path,
    metric_name: str,
) -> dict[str, list[tuple[float, float]]]:
    if filepath.suffix == ".csv":
        return _load_csv(filepath, metric_name)
    if filepath.suffix == ".json":
        return _load_json(filepath, metric_name)
    raise ValueError(
        f"Unsupported file type: {filepath.suffix} (expected .csv or .json)"
    )


def parse_source_args(source_args: list[str]) -> dict[str, list[Path]]:
    """Parse ``LABEL=PATH`` strings into ``{label: [path, ...]}``."""
    result: dict[str, list[Path]] = defaultdict(list)
    for arg in source_args:
        if "=" not in arg:
            raise ValueError(f"Invalid source format: '{arg}'. Expected 'LABEL=PATH'.")
        label, path_str = arg.split("=", 1)
        path = Path(path_str.strip())
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        result[label.strip()].append(path)
    return dict(result)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_curve(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    n_dense: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a smooth spline through noisy data."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    if len(x) < 2:  # noqa: PLR2004
        return x, y

    spline = UnivariateSpline(x, y, s=len(x) * np.var(y))
    x_dense = np.linspace(x.min(), x.max(), n_dense)
    y_dense = spline(x_dense)

    return x_dense, y_dense


def pretty_subset(name: str) -> str:
    return PRETTY_SUBSET_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Gen-len parsing
# ---------------------------------------------------------------------------


def _extract_subset(filepath: Path) -> str:
    """Extract subset name from gen_len_*.json or sweep(s)_*.json filenames."""
    match = re.search(r"(?:gen_len|sweeps?)_(.+)\.json$", filepath.name)
    return match.group(1) if match else filepath.stem


def parse_gen_len_file(filepath: Path) -> dict:
    """Parse a single gen-len JSON and return statistics."""
    with filepath.open() as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise ValueError(f"No benchmarks found in {filepath}")

    successful = benchmarks[0].get("requests", {}).get("successful", [])
    if not successful:
        raise ValueError(f"No successful requests found in {filepath}")

    output_tokens = [r["output_tokens"] for r in successful]
    median = statistics.median(output_tokens)

    return {
        "count": len(output_tokens),
        "median": median,
        "min": min(output_tokens),
        "max": max(output_tokens),
        "max_tokens": 2 ** math.ceil(math.log2(max(median, 1))),
    }


# ---------------------------------------------------------------------------
# Prometheus metrics parsing
# ---------------------------------------------------------------------------

_RE_SIMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)$")
_RE_LABELED = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]+)\}\s+([0-9.eE+-]+)$")
_SPEC = "vllm:spec_decode"


def parse_prometheus_metrics(raw_text: str) -> list[Metric]:
    """Parse Prometheus-formatted metrics into Metric objects."""
    if not raw_text:
        return []

    metrics: list[Metric] = []
    vector_data: dict[str, list[tuple[int, float]]] = {}

    for raw_line in raw_text.strip().split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        m = _RE_SIMPLE.match(line) or _RE_LABELED.match(line)
        if not m or _SPEC not in m.group(1):
            continue

        groups = m.groups()
        name = groups[0].replace("_total", "")
        value = float(groups[-1])

        if "per_pos" not in name:
            metrics.append(Counter(name=name, value=value))
        elif len(groups) == 3:  # noqa: PLR2004
            pos_match = re.search(r'position="(\d+)"', groups[1])
            if pos_match:
                pos = int(pos_match.group(1))
                vector_data.setdefault(name, []).append((pos, value))

    for name, pos_values in vector_data.items():
        pos_values.sort()
        values = [0.0] * (pos_values[-1][0] + 1)
        for pos, val in pos_values:
            values[pos] = val
        metrics.append(Vector(name=name, values=values))

    return metrics


def _accumulate(metrics: list[Metric]) -> tuple[float, float, float, list[float]]:
    sums: dict[str, float] = defaultdict(float)
    counts: list[float] = []
    for m in metrics:
        if isinstance(m, Counter):
            sums[m.name] += m.value
        elif isinstance(m, Vector) and "per_pos" in m.name:
            counts = [a + b for a, b in zip_longest(counts, m.values, fillvalue=0.0)]
    return (
        sums.get("vllm:spec_decode_num_drafts", 0),
        sums.get("vllm:spec_decode_num_draft_tokens", 0),
        sums.get("vllm:spec_decode_num_accepted_tokens", 0),
        counts,
    )


def extract_spec_decode_metrics(
    raw_metrics: list[Metric],
    baseline_metrics: list[Metric] | None = None,
) -> dict[str, float]:
    """Extract speculative decoding metrics and calculate acceptance rates."""
    drafts, draft_tok, accepted, counts = _accumulate(raw_metrics)

    if baseline_metrics:
        bd, bdt, ba, bc = _accumulate(baseline_metrics)
        drafts -= bd
        draft_tok -= bdt
        accepted -= ba
        counts = [a - b for a, b in zip_longest(counts, bc, fillvalue=0.0)]

    result: dict[str, float] = {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tok,
        "num_accepted_tokens": accepted,
        "acceptance_length": 1 + accepted / drafts if drafts > 0 else 0,
    }
    for i, count in enumerate(counts):
        result[f"acceptance_at_pos_{i}"] = count / drafts if drafts > 0 else 0
    return result


def load_prometheus_file(path: Path | None) -> list[Metric]:
    if path and path.exists():
        with path.open() as f:
            return parse_prometheus_metrics(f.read())
    return []


# ---------------------------------------------------------------------------
# Sweep parsing
# ---------------------------------------------------------------------------


def parse_sweep_file(filepath: Path) -> list[dict]:
    """Parse a single sweep JSON and return rows for the CSV."""
    with filepath.open() as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise ValueError(f"No benchmarks found in {filepath}")

    subset = _extract_subset(filepath)
    rows = []
    for bm in benchmarks:
        strategy = bm.get("config", {}).get("strategy", {})
        strategy_type = strategy.get("type_", "unknown")
        if strategy_type in SKIP_STRATEGIES:
            continue

        metrics = bm.get("metrics", {})
        row = {
            "subset": subset,
            "strategy": strategy_type,
            "target_rate": strategy.get("rate", ""),
        }
        for metric_key, csv_key in METRICS_TO_EXTRACT:
            val = metrics.get(metric_key, {})
            row[csv_key] = val.get("successful", {}).get("median", "")
        out_tok = metrics.get("output_tokens", {})
        row["total_output_tokens"] = out_tok.get("successful", {}).get("sum", "")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Higher-level result writers
# ---------------------------------------------------------------------------


def acceptance_csv_columns(spec: dict[str, float]) -> list[str]:
    cols = [
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "acceptance_length",
    ]
    pos = 0
    while f"acceptance_at_pos_{pos}" in spec:
        cols.append(f"acceptance_at_pos_{pos}")
        pos += 1
    return cols


def parse_gen_len_results(files: list[Path], output: Path) -> dict[str, int]:
    """Parse gen-len JSONs into a {subset: max_tokens} mapping and write to *output*."""
    header = (
        f"{'Subset':<20} {'Count':>6} {'Median':>8}"
        f" {'Min':>8} {'Max':>8} {'max_tokens':>12}"
    )
    print(header)
    print("-" * 70)

    max_tokens_map: dict[str, int] = {}
    for filepath in files:
        subset = _extract_subset(filepath)
        try:
            s = parse_gen_len_file(filepath)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning("Skipping %s: %s", filepath, e)
            continue
        max_tokens_map[subset] = s["max_tokens"]
        print(
            f"{subset:<20} {s['count']:>6} {s['median']:>8.0f}"
            f" {s['min']:>8} {s['max']:>8} {s['max_tokens']:>12}"
        )

    if not max_tokens_map:
        logger.error("No files were successfully parsed")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(max_tokens_map, f, indent=2)
    print(f"\nmax_tokens mapping written to: {output}")
    return max_tokens_map


def parse_sweep_results(
    filepath: Path,
    spec_decode_metrics: dict[str, float] | None = None,
) -> list[dict]:
    """Parse a sweep JSON and enrich rows with pre-computed acceptance metrics."""
    rows = parse_sweep_file(filepath)
    if spec_decode_metrics:
        for row in rows:
            row.update(spec_decode_metrics)
    return rows


class CsvWriter:
    """Append rows incrementally to a CSV file, writing the header on first row."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        self._started = self.path.exists()

    def append(self, row: dict) -> None:
        self.append_rows([row])

    def append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._started else "w"
        with self.path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            if not self._started:
                writer.writeheader()
                self._started = True
            writer.writerows(rows)


def print_acceptance_report(spec: dict[str, float]) -> None:
    summary = [
        ("Num drafts", f"{spec.get('num_drafts', 0):.0f}"),
        ("Num draft tokens", f"{spec.get('num_draft_tokens', 0):.0f}"),
        ("Num accepted tokens", f"{spec.get('num_accepted_tokens', 0):.0f}"),
        ("Acceptance length", f"{spec.get('acceptance_length', 0):.4f}"),
    ]
    print("\n=== Speculative Decoding Acceptance Report ===\n")
    for label, val in summary:
        print(f"  {label:<25} {val:>12}")

    positions = [
        (i, spec[f"acceptance_at_pos_{i}"])
        for i in range(100)
        if f"acceptance_at_pos_{i}" in spec
    ]
    if positions:
        print(f"\n  {'Position':<25} {'Acceptance Rate':>12}")
        print(f"  {'-' * 25} {'-' * 12}")
        for pos, rate in positions:
            print(f"  {f'Position {pos}':<25} {rate:>12.4f}")
    print()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def check_dependencies() -> None:
    missing = [cmd for cmd in ("guidellm", "python") if not shutil.which(cmd)]
    if missing:
        logger.error("Missing required dependencies: %s", ", ".join(missing))
        sys.exit(1)


def build_backend_args(gen_kwargs: str, max_tokens: int) -> str:
    if gen_kwargs:
        try:
            body = json.loads(gen_kwargs)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in --gen-kwargs: {gen_kwargs!r}: {e}"
            raise ValueError(msg) from e
    else:
        body = {}
    body["max_tokens"] = max_tokens
    return json.dumps({"extras": {"body": body}})


def fetch_metrics(metrics_url: str, retries: int = 3, delay: float = 2.0) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            with urlopen(metrics_url, timeout=30) as resp:  # noqa: S310
                return resp.read().decode()
        except (URLError, OSError) as e:
            logger.warning(
                "Failed to fetch metrics from %s (attempt %d/%d): %s",
                metrics_url,
                attempt,
                retries,
                e,
            )
            if attempt < retries:
                time.sleep(delay)
    return None


def run_guidellm(
    target: str,
    dataset: str,
    subset: str,
    data_column_mapper: str,
    profile: str,
    rate: int | str,
    max_requests: int | None,
    max_concurrency: int,
    output_path: Path,
    backend_args: str,
) -> None:
    cmd = [
        "guidellm",
        "benchmark",
        "--target",
        target,
        "--data",
        dataset,
        "--data-args",
        json.dumps({"data_files": f"{subset}.jsonl"}),
        "--data-column-mapper",
        data_column_mapper,
        "--profile",
        profile,
        "--rate",
        str(rate),
        "--output-path",
        str(output_path),
        "--backend-args",
        backend_args,
    ]
    if max_requests is not None:
        cmd.extend(["--max-requests", str(max_requests)])

    env = os.environ.copy()
    env["GUIDELLM__MAX_CONCURRENCY"] = str(max_concurrency)
    subprocess.run(cmd, check=True, env=env)  # noqa: S603
