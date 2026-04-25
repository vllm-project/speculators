"""Shared utilities for performance visualization scripts.

Provides data loading (CSV + GuideLLM JSON), metric definitions,
smoothing functions (binning + isotonic regression + PCHIP), and
CLI argument helpers.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

_MIN_POINTS_FOR_INTERP = 2

# ---------------------------------------------------------------------------
# Metric definitions
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
    "acceptance_rate": {
        "csv_col": "acceptance_rate",
        "json_key": None,
        "stat": None,
        "label": "Acceptance Rate",
        "increasing": False,
    },
    "mean_accepted": {
        "csv_col": "mean_accepted_tokens",
        "json_key": None,
        "stat": None,
        "label": "Mean Accepted Tokens",
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_csv(
    filepath: Path,
    metric_name: str,
) -> dict[str, list[tuple[float, float]]]:
    """Load constant-rate data points from a CSV file."""
    metric_cfg = METRICS[metric_name]
    csv_col = metric_cfg["csv_col"]

    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with filepath.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("strategy") != "constant":
                continue
            try:
                rps = float(row["rps_median"])
                y = float(row[csv_col])
            except (ValueError, KeyError):
                continue
            subset = row.get("subset", "unknown")
            result[subset].append((rps, y))

    return dict(result)


def _load_json(
    filepath: Path,
    metric_name: str,
) -> dict[str, list[tuple[float, float]]]:
    """Load constant-rate data points from a GuideLLM sweep JSON."""
    metric_cfg = METRICS[metric_name]
    json_key = metric_cfg["json_key"]
    stat = metric_cfg["stat"]

    with filepath.open() as f:
        data = json.load(f)

    subset = _extract_subset_from_json(data)
    points: list[tuple[float, float]] = []

    for bench in data.get("benchmarks", []):
        strategy_type = bench.get("config", {}).get("strategy", {}).get("type_")
        if strategy_type != "constant":
            continue
        try:
            rps = bench["metrics"]["requests_per_second"]["successful"]["mean"]
            y = bench["metrics"][json_key]["successful"][stat]
            points.append((rps, y))
        except (KeyError, TypeError):
            continue

    points.sort(key=lambda p: p[0])
    return {subset: points} if points else {}


def _extract_subset_from_json(data: dict) -> str:
    """Extract subset name from the JSON content's data_args field."""
    data_files = data["args"]["data_args"][0]["data_files"]
    return Path(data_files).stem


def load_data(
    filepath: Path,
    metric_name: str,
) -> dict[str, list[tuple[float, float]]]:
    """Load data from a CSV or JSON file, auto-detected by extension.

    Returns ``{subset_name: [(rps, y_value), ...]}``.
    """
    if filepath.suffix == ".csv":
        return _load_csv(filepath, metric_name)
    elif filepath.suffix == ".json":
        return _load_json(filepath, metric_name)
    else:
        raise ValueError(
            f"Unsupported file type: {filepath.suffix} (expected .csv or .json)"
        )


# ---------------------------------------------------------------------------
# Source argument parsing
# ---------------------------------------------------------------------------


def parse_source_args(source_args: list[str]) -> dict[str, list[Path]]:
    """Parse ``LABEL=PATH`` strings into ``{label: [path, ...]}``.

    Multiple entries with the same label are grouped (for pooling repetitions).
    """
    result: dict[str, list[Path]] = defaultdict(list)
    for arg in source_args:
        if "=" not in arg:
            raise ValueError(f"Invalid source format: '{arg}'. Expected 'LABEL=PATH'.")
        label, path_str = arg.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        result[label].append(path)
    return dict(result)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def isotonic(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Pool-adjacent-violators algorithm for isotonic regression."""
    y = np.array(y, dtype=float)
    if not increasing:
        return -isotonic(-y, increasing=True)

    n = len(y)
    target = y.copy()
    weight = np.ones(n, dtype=float)
    blocks = list(range(n))

    i = 0
    while i < len(blocks) - 1:
        cur = blocks[i]
        nxt = blocks[i + 1]
        if target[cur] > target[nxt]:
            w_sum = weight[cur] + weight[nxt]
            target[cur] = (
                weight[cur] * target[cur] + weight[nxt] * target[nxt]
            ) / w_sum
            weight[cur] = w_sum
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.empty(n, dtype=float)
    prev = 0
    for b in blocks:
        end = blocks[blocks.index(b) + 1] if b != blocks[-1] else n
        out[prev:end] = target[b]
        prev = end
    return out


def smooth_curve(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    n_dense: int = 200,
    increasing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin, apply isotonic regression, then PCHIP-interpolate.

    Returns ``(x_dense, y_dense)`` arrays of length *n_dense*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    if len(x) < _MIN_POINTS_FOR_INTERP:
        return x, y

    x_min, x_max = x.min(), x.max()

    n_bins = max(min(len(x) // 3, 15), 4)
    edges = np.linspace(x_min, x_max, n_bins + 1)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (x >= lo) & (x < hi) if hi < edges[-1] else (x >= lo) & (x <= hi)
        if mask.any():
            bx.append(x[mask].mean())
            by.append(y[mask].mean())
    bx, by = np.array(bx), np.array(by)

    by = isotonic(by, increasing=increasing)

    _, unique_idx = np.unique(bx, return_index=True)
    bx, by = bx[unique_idx], by[unique_idx]

    if len(bx) < _MIN_POINTS_FOR_INTERP:
        return bx, by

    interp = PchipInterpolator(bx, by)
    x_dense = np.linspace(x_min, x_max, n_dense)
    y_dense = interp(x_dense)
    if increasing:
        y_dense = np.maximum.accumulate(y_dense)
    else:
        y_dense = np.minimum.accumulate(y_dense)
    return x_dense, y_dense


def pretty_subset(name: str) -> str:
    """Return a human-readable name for a subset."""
    return PRETTY_SUBSET_NAMES.get(name, name)
