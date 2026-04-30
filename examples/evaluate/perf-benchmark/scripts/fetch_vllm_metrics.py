#!/usr/bin/env python3
"""Fetch vLLM metrics and calculate speculative decoding acceptance rates.

Fetches metrics from a vLLM server's /metrics endpoint, parses the
Prometheus-formatted output, and calculates per-position acceptance rates
for speculative decoding.

Usage:
    python fetch_vllm_metrics.py --url http://localhost:8000 --output metrics.json
    python fetch_vllm_metrics.py --url http://localhost:8000 --total-tokens 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

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


def fetch_metrics(url: str, timeout: int = 10) -> str:
    """Fetch raw metrics from vLLM /metrics endpoint."""
    metrics_url = url.rstrip("/") + "/metrics"
    try:
        response = requests.get(metrics_url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch metrics from %s: %s", metrics_url, e)
        sys.exit(1)


def parse_prometheus_metrics(raw_text: str) -> list[Metric]:  # noqa: C901
    """Parse Prometheus-formatted metrics into Metric objects."""
    metrics: list[Metric] = []
    lines = raw_text.strip().split("\n")

    # Track vector metrics (those with position labels)
    vector_data: dict[str, list[tuple[int, float]]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Match metric lines like: metric_name{labels} value
        # or simple: metric_name value
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)$", line)
        if match:
            name, value = match.groups()
            if "vllm:spec_decode" in name:
                metrics.append(Counter(name=name, value=float(value)))
            continue

        # Match labeled metrics like: metric_name{label="value"} value
        match = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]+)\}\s+([0-9.eE+-]+)$", line
        )
        if match:
            name, labels_str, value = match.groups()
            if "vllm:spec_decode_num_accepted_tokens_per_pos" in name:
                # Extract position from labels
                pos_match = re.search(r'position="(\d+)"', labels_str)
                if pos_match:
                    position = int(pos_match.group(1))
                    if name not in vector_data:
                        vector_data[name] = []
                    vector_data[name].append((position, float(value)))

    # Convert vector data to Vector metrics
    for name, pos_values in vector_data.items():
        pos_values.sort(key=lambda x: x[0])  # Sort by position
        max_pos = pos_values[-1][0] if pos_values else 0
        values = [0.0] * (max_pos + 1)
        for pos, val in pos_values:
            values[pos] = val
        metrics.append(Vector(name=name, values=values))

    return metrics


def extract_metrics(  # noqa: C901
    raw_metrics: list[Metric], total_num_output_tokens: int
) -> dict:
    """Extract speculative decoding metrics and calculate acceptance rates."""
    metrics_dict: dict[str, int | float] = {}
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts: list[float] = []

    for metric in raw_metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            if len(acceptance_counts) < len(metric.values):
                acceptance_counts = acceptance_counts + [0.0] * (
                    len(metric.values) - len(acceptance_counts)
                )
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    metrics_dict["total_num_output_tokens"] = total_num_output_tokens
    metrics_dict["num_drafts"] = num_drafts
    metrics_dict["num_draft_tokens"] = num_draft_tokens
    metrics_dict["num_accepted_tokens"] = num_accepted_tokens
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    metrics_dict["acceptance_length"] = acceptance_length

    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        metrics_dict[f"acceptance_at_token_{i}"] = acceptance_rate

    return metrics_dict


def format_output(metrics: dict[str, int | float]) -> str:
    """Format metrics for human-readable output."""
    lines = []
    lines.append("=== Speculative Decoding Metrics ===")
    lines.append(f"Total output tokens: {metrics.get('total_num_output_tokens', 0)}")
    lines.append(f"Number of drafts: {metrics.get('num_drafts', 0)}")
    lines.append(f"Draft tokens proposed: {metrics.get('num_draft_tokens', 0)}")
    lines.append(f"Draft tokens accepted: {metrics.get('num_accepted_tokens', 0)}")
    lines.append(
        f"Average acceptance length: {metrics.get('acceptance_length', 0):.2f}"
    )
    lines.append("\nPer-position acceptance rates:")

    pos = 0
    while f"acceptance_at_token_{pos}" in metrics:
        rate = metrics[f"acceptance_at_token_{pos}"]
        lines.append(f"  Position {pos}: {rate:.4f} ({rate * 100:.2f}%)")
        pos += 1

    return "\n".join(lines)


def main() -> None:
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Fetch vLLM metrics and calculate acceptance rates."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=0,
        help="Total number of output tokens generated (default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Fetch raw metrics
    logger.info("Fetching metrics from %s/metrics", args.url)
    raw_text = fetch_metrics(args.url, timeout=args.timeout)

    # Parse metrics
    parsed_metrics = parse_prometheus_metrics(raw_text)
    logger.info("Parsed %d speculative decoding metrics", len(parsed_metrics))

    # Extract and calculate
    result = extract_metrics(parsed_metrics, args.total_tokens)

    # Output
    logger.info("\n%s", format_output(result))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2)
        logger.info("Metrics saved to: %s", args.output)


if __name__ == "__main__":
    main()
