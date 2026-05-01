#!/usr/bin/env python3
"""Parse guidellm sweep output files and extract performance metrics to CSV.

Reads one or more sweep JSON files and extracts key metrics for each
sweep point (synchronous and constant-rate). Throughput-mode entries
are excluded as they represent max-load saturation.

Optionally enriches results with vLLM speculative decoding metrics.

Usage:
    # Basic performance metrics only
    python parse_sweep.py --output results.csv sweep_*.json

    # With vLLM speculative decoding metrics
    python parse_sweep.py --output results.csv \
        --current-metrics current.txt \
        --baseline-metrics baseline.txt \
        sweep_*.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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


def parse_prometheus_metrics(raw_text: str) -> list[Metric]:  # noqa: C901
    """Parse Prometheus-formatted metrics into Metric objects."""
    if not raw_text:
        return []

    metrics: list[Metric] = []
    lines = raw_text.strip().split("\n")
    vector_data: dict[str, list[tuple[int, float]]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Match simple counter: metric_name value
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)$", line)
        if match:
            name, value = match.groups()
            if "vllm:spec_decode" in name:
                name = name.replace("_total", "")
                metrics.append(Counter(name=name, value=float(value)))
            continue

        # Match labeled metrics: metric_name{labels} value
        match = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]+)\}\s+([0-9.eE+-]+)$", line
        )
        if match:
            name, labels_str, value = match.groups()
            if "vllm:spec_decode" in name and "per_pos" not in name:
                name = name.replace("_total", "")
                metrics.append(Counter(name=name, value=float(value)))
            elif "vllm:spec_decode_num_accepted_tokens_per_pos" in name:
                pos_match = re.search(r'position="(\d+)"', labels_str)
                if pos_match:
                    position = int(pos_match.group(1))
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


def extract_spec_decode_metrics(  # noqa: C901
    raw_metrics: list[Metric], baseline_metrics: list[Metric] | None = None
) -> dict[str, float]:
    """Extract speculative decoding metrics and calculate acceptance rates."""

    def _accumulate_metrics(
        metrics: list[Metric],
    ) -> tuple[float, float, float, list[float]]:
        """Helper to accumulate raw counter values."""
        num_drafts = 0.0
        num_draft_tokens = 0.0
        num_accepted_tokens = 0.0
        acceptance_counts: list[float] = []

        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_drafts" and isinstance(
                metric, Counter
            ):
                num_drafts += metric.value
            elif metric.name == "vllm:spec_decode_num_draft_tokens" and isinstance(
                metric, Counter
            ):
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(
                metric, Counter
            ):
                num_accepted_tokens += metric.value
            elif (
                metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
                and isinstance(metric, Vector)
            ):
                if len(acceptance_counts) < len(metric.values):
                    acceptance_counts = acceptance_counts + [0.0] * (
                        len(metric.values) - len(acceptance_counts)
                    )
                for pos in range(len(metric.values)):
                    acceptance_counts[pos] += metric.values[pos]

        return num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts

    # Accumulate current metrics
    num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts = (
        _accumulate_metrics(raw_metrics)
    )

    # Subtract baseline if provided (for delta)
    if baseline_metrics:
        base_drafts, base_draft_tokens, base_accepted_tokens, base_acceptance_counts = (
            _accumulate_metrics(baseline_metrics)
        )
        num_drafts -= base_drafts
        num_draft_tokens -= base_draft_tokens
        num_accepted_tokens -= base_accepted_tokens

        if len(base_acceptance_counts) < len(acceptance_counts):
            base_acceptance_counts += [0.0] * (
                len(acceptance_counts) - len(base_acceptance_counts)
            )

        for pos in range(len(acceptance_counts)):
            acceptance_counts[pos] -= (
                base_acceptance_counts[pos]
                if pos < len(base_acceptance_counts)
                else 0.0
            )

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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Parse sweep results and extract metrics to CSV."
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
        "--current-metrics",
        type=Path,
        help="File containing current vLLM metrics (enables spec decode metrics)",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        help="File containing baseline vLLM metrics to subtract (for computing deltas)",
    )
    args = parser.parse_args()

    # Load baseline metrics if provided
    baseline_metrics: list[Metric] = []
    if args.baseline_metrics:
        if args.baseline_metrics.exists():
            with args.baseline_metrics.open() as f:
                baseline_text = f.read()
            baseline_metrics = parse_prometheus_metrics(baseline_text)
            logger.info("Loaded baseline metrics from %s", args.baseline_metrics)
        else:
            logger.warning("Baseline metrics file not found: %s", args.baseline_metrics)

    # Parse sweep files
    all_rows: list[dict] = []
    total_output_tokens = 0

    for filepath in args.files:
        if not filepath.exists():
            logger.warning("File not found, skipping: %s", filepath)
            continue

        try:
            rows = parse_sweep_file(filepath)
            all_rows.extend(rows)
            # Sum up total output tokens
            for row in rows:
                if row.get("total_output_tokens"):
                    with contextlib.suppress(ValueError, TypeError):
                        total_output_tokens += int(row["total_output_tokens"])
            logger.info("Parsed %d entries from %s", len(rows), filepath.name)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning("Failed to parse %s: %s", filepath, e)
            continue

    if not all_rows:
        logger.error("No data was successfully parsed")
        sys.exit(1)

    # Fetch vLLM speculative decoding metrics if current metrics file provided
    spec_decode_metrics = {}
    if args.current_metrics:
        if args.current_metrics.exists():
            logger.info("Reading current metrics from %s", args.current_metrics)
            with args.current_metrics.open() as f:
                raw_text = f.read()

            parsed_metrics = parse_prometheus_metrics(raw_text)
            spec_decode_metrics = extract_spec_decode_metrics(
                parsed_metrics,
                baseline_metrics=baseline_metrics if baseline_metrics else None,
            )
            logger.info("Extracted %d spec decode metrics", len(spec_decode_metrics))

            if spec_decode_metrics:
                logger.info(
                    "Acceptance length: %.2f",
                    spec_decode_metrics.get("acceptance_length", 0),
                )
                logger.info("Total output tokens: %d", total_output_tokens)
                logger.info(
                    "Num drafts: %.0f", spec_decode_metrics.get("num_drafts", 0)
                )
        else:
            logger.warning("Current metrics file not found: %s", args.current_metrics)

    # Build CSV columns dynamically
    csv_columns = BASE_CSV_COLUMNS.copy()
    if spec_decode_metrics:
        csv_columns.extend(
            [
                "num_drafts",
                "num_draft_tokens",
                "num_accepted_tokens",
                "acceptance_length",
            ]
        )
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
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(
        "\nCSV written to: %s (%d rows, %d columns)",
        args.output,
        len(all_rows),
        len(csv_columns),
    )


if __name__ == "__main__":
    main()
