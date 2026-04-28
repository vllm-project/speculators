#!/usr/bin/env python3
"""Parse guidellm sweep results and enhance with vLLM speculative decoding metrics.

Combines performance metrics from guidellm sweep JSON files with speculative
decoding acceptance rates from vLLM's /metrics endpoint.

Usage:
    python parse_sweep_with_metrics.py --output results.csv --vllm-url http://localhost:8000 sweep_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class Metric:
    """Base metric class."""

    name: str


@dataclass
class Counter(Metric):
    """Counter metric with a single value."""

    value: float


@dataclass
class Vector(Metric):
    """Vector metric with multiple labeled values."""

    values: list[float]


# Performance metrics to extract from guidellm
METRICS_TO_EXTRACT = [
    ("requests_per_second", "rps_median"),
    ("request_latency", "latency_median_s"),
    ("inter_token_latency_ms", "itl_median_ms"),
    ("time_to_first_token_ms", "ttft_median_ms"),
    ("output_tokens_per_second", "output_tps_median"),
]

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

SKIP_STRATEGIES = {"throughput"}


def fetch_vllm_metrics(url: str, timeout: int = 10) -> str:
    """Fetch raw metrics from vLLM /metrics endpoint."""
    metrics_url = url.rstrip("/") + "/metrics"
    try:
        response = requests.get(metrics_url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Failed to fetch vLLM metrics from {metrics_url}: {e}", file=sys.stderr)
        return ""


def parse_prometheus_metrics(raw_text: str) -> list[Metric]:
    """Parse Prometheus-formatted metrics into Metric objects."""
    if not raw_text:
        return []

    metrics: list[Metric] = []
    lines = raw_text.strip().split("\n")
    vector_data: dict[str, list[tuple[int, float]]] = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match simple counter: metric_name value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)$', line)
        if match:
            name, value = match.groups()
            if "vllm:spec_decode" in name:
                # Remove _total suffix for compatibility
                name = name.replace("_total", "")
                metrics.append(Counter(name=name, value=float(value)))
            continue

        # Match labeled metrics: metric_name{labels} value
        match = re.match(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]+)\}\s+([0-9.eE+-]+)$', line
        )
        if match:
            name, labels_str, value = match.groups()
            # Handle labeled spec decode counter metrics
            if "vllm:spec_decode" in name and "per_pos" not in name:
                # Remove _total suffix for compatibility
                name = name.replace("_total", "")
                metrics.append(Counter(name=name, value=float(value)))
            # Handle per-position vector metrics
            elif "vllm:spec_decode_num_accepted_tokens_per_pos" in name:
                pos_match = re.search(r'position="(\d+)"', labels_str)
                if pos_match:
                    position = int(pos_match.group(1))
                    # Remove _total suffix for compatibility
                    base_name = name.replace("_total", "")
                    if base_name not in vector_data:
                        vector_data[base_name] = []
                    vector_data[base_name].append((position, float(value)))

    # Convert vector data to Vector metrics
    for name, pos_values in vector_data.items():
        pos_values.sort(key=lambda x: x[0])
        max_pos = pos_values[-1][0] if pos_values else 0
        values = [0.0] * (max_pos + 1)
        for pos, val in pos_values:
            values[pos] = val
        metrics.append(Vector(name=name, values=values))

    return metrics


def extract_spec_decode_metrics(raw_metrics: list[Metric], baseline_metrics: list[Metric] | None = None) -> dict[str, float]:
    """Extract speculative decoding metrics and calculate acceptance rates.

    Args:
        raw_metrics: Current metrics snapshot
        baseline_metrics: Optional baseline snapshot to subtract (for computing deltas)

    Returns:
        Dictionary of computed metrics. If baseline provided, returns the delta.
    """
    def _accumulate_metrics(metrics: list[Metric]) -> tuple[float, float, float, list[float]]:
        """Helper to accumulate raw counter values."""
        num_drafts = 0.0
        num_draft_tokens = 0.0
        num_accepted_tokens = 0.0
        acceptance_counts: list[float] = []

        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_drafts":
                assert isinstance(metric, Counter)
                num_drafts += metric.value
            elif metric.name == "vllm:spec_decode_num_draft_tokens":
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                assert isinstance(metric, Vector)
                if len(acceptance_counts) < len(metric.values):
                    acceptance_counts = acceptance_counts + [0.0] * (
                        len(metric.values) - len(acceptance_counts)
                    )
                for pos in range(len(metric.values)):
                    acceptance_counts[pos] += metric.values[pos]

        return num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts

    # Accumulate current metrics
    num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts = _accumulate_metrics(raw_metrics)

    # Subtract baseline if provided (for delta)
    if baseline_metrics:
        base_drafts, base_draft_tokens, base_accepted_tokens, base_acceptance_counts = _accumulate_metrics(baseline_metrics)
        num_drafts -= base_drafts
        num_draft_tokens -= base_draft_tokens
        num_accepted_tokens -= base_accepted_tokens

        # Ensure baseline acceptance_counts has same length
        if len(base_acceptance_counts) < len(acceptance_counts):
            base_acceptance_counts += [0.0] * (len(acceptance_counts) - len(base_acceptance_counts))

        for pos in range(len(acceptance_counts)):
            acceptance_counts[pos] -= base_acceptance_counts[pos] if pos < len(base_acceptance_counts) else 0.0

    metrics_dict: dict[str, float] = {}
    metrics_dict["num_drafts"] = num_drafts
    metrics_dict["num_draft_tokens"] = num_draft_tokens
    metrics_dict["num_accepted_tokens"] = num_accepted_tokens
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 0
    metrics_dict["acceptance_length"] = acceptance_length

    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        metrics_dict[f"acceptance_at_pos_{i}"] = acceptance_rate

    return metrics_dict


def extract_subset_name(filepath: Path) -> str:
    """Derive subset name from filename like sweep_HumanEval.json."""
    match = re.search(r"sweep[s]?_(.+)\.json$", filepath.name)
    if match:
        return match.group(1)
    return filepath.stem


def parse_sweep_file(filepath: Path) -> list[dict]:
    """Parse a single sweep JSON and return rows for the CSV."""
    with filepath.open() as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise ValueError(f"No benchmarks found in {filepath}")

    subset = extract_subset_name(filepath)
    rows = []

    for bm in benchmarks:
        config = bm.get("config", {})
        strategy = config.get("strategy", {})
        strategy_type = strategy.get("type_", "unknown")

        if strategy_type in SKIP_STRATEGIES:
            continue

        target_rate = strategy.get("rate", "")
        metrics = bm.get("metrics", {})

        row = {
            "subset": subset,
            "strategy": strategy_type,
            "target_rate": target_rate if target_rate else "",
        }

        # Extract performance metrics
        for metric_key, csv_key in METRICS_TO_EXTRACT:
            metric_data = metrics.get(metric_key, {})
            successful = metric_data.get("successful", {})
            row[csv_key] = successful.get("median", "")

        # Extract total output tokens
        output_token_data = metrics.get("output_tokens", {}).get("successful", {})
        row["total_output_tokens"] = output_token_data.get("sum", "")

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse sweep results and enhance with vLLM spec decode metrics."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Sweep JSON files (e.g. sweep_HumanEval.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        help="vLLM server base URL (e.g. http://localhost:8000) to fetch spec decode metrics",
    )
    parser.add_argument(
        "--baseline-metrics-file",
        type=Path,
        help="File containing baseline vLLM metrics to subtract (for computing deltas)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout for vLLM metrics request in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Load baseline metrics if provided
    baseline_metrics: list[Metric] = []
    if args.baseline_metrics_file:
        if args.baseline_metrics_file.exists():
            with args.baseline_metrics_file.open() as f:
                baseline_text = f.read()
            baseline_metrics = parse_prometheus_metrics(baseline_text)
            print(f"[INFO] Loaded baseline metrics from {args.baseline_metrics_file}")
        else:
            print(f"[WARN] Baseline metrics file not found: {args.baseline_metrics_file}", file=sys.stderr)

    # Parse sweep files
    all_rows: list[dict] = []
    total_output_tokens = 0

    for filepath in args.files:
        if not filepath.exists():
            print(f"[WARN] File not found, skipping: {filepath}", file=sys.stderr)
            continue

        try:
            rows = parse_sweep_file(filepath)
            all_rows.extend(rows)
            # Sum up total output tokens
            for row in rows:
                if row.get("total_output_tokens"):
                    try:
                        total_output_tokens += int(row["total_output_tokens"])
                    except (ValueError, TypeError):
                        pass
            print(f"[INFO] Parsed {len(rows)} entries from {filepath.name}")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to parse {filepath}: {e}", file=sys.stderr)
            continue

    if not all_rows:
        print("[ERROR] No data was successfully parsed", file=sys.stderr)
        sys.exit(1)

    # Fetch vLLM speculative decoding metrics if URL provided
    spec_decode_metrics = {}
    if args.vllm_url:
        print(f"[INFO] Fetching vLLM metrics from {args.vllm_url}")
        raw_text = fetch_vllm_metrics(args.vllm_url, timeout=args.timeout)
        if raw_text:
            parsed_metrics = parse_prometheus_metrics(raw_text)
            # Compute delta if baseline provided, otherwise use raw metrics
            spec_decode_metrics = extract_spec_decode_metrics(
                parsed_metrics,
                baseline_metrics=baseline_metrics if baseline_metrics else None
            )
            print(f"[INFO] Extracted {len(spec_decode_metrics)} spec decode metrics")

            # Print summary
            if spec_decode_metrics:
                print(f"[INFO] Acceptance length: {spec_decode_metrics.get('acceptance_length', 0):.2f}")
                print(f"[INFO] Total output tokens: {total_output_tokens}")
                print(f"[INFO] Num drafts: {spec_decode_metrics.get('num_drafts', 0):.0f}")

    # Build CSV columns dynamically based on available spec decode metrics
    csv_columns = BASE_CSV_COLUMNS.copy()
    if spec_decode_metrics:
        # Add acceptance metrics columns
        csv_columns.extend(["num_drafts", "num_draft_tokens", "num_accepted_tokens", "acceptance_length"])
        # Add per-position acceptance rates
        pos = 0
        while f"acceptance_at_pos_{pos}" in spec_decode_metrics:
            csv_columns.append(f"acceptance_at_pos_{pos}")
            pos += 1

    # Add spec decode metrics to all rows (they're global)
    if spec_decode_metrics:
        for row in all_rows:
            for key, value in spec_decode_metrics.items():
                row[key] = value

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[INFO] CSV written to: {args.output} ({len(all_rows)} rows, {len(csv_columns)} columns)")


if __name__ == "__main__":
    main()
