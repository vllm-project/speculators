#!/usr/bin/env python3
"""Parse guidellm sweep output files and extract performance metrics to CSV.

Reads one or more sweep JSON files and extracts key metrics for each
sweep point (synchronous and constant-rate). Throughput-mode entries
are excluded as they represent max-load saturation.

Usage:
    python parse_sweep_results.py --output results.csv sweep_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

METRICS_TO_EXTRACT = [
    ("requests_per_second", "rps_median"),
    ("request_latency", "latency_median_s"),
    ("inter_token_latency_ms", "itl_median_ms"),
    ("time_to_first_token_ms", "ttft_median_ms"),
    ("output_tokens_per_second", "output_tps_median"),
]

CSV_COLUMNS = [
    "subset",
    "strategy",
    "target_rate",
    "rps_median",
    "latency_median_s",
    "itl_median_ms",
    "ttft_median_ms",
    "output_tps_median",
]

SKIP_STRATEGIES = {"throughput"}


def extract_subset_name(filepath: Path) -> str:
    """Derive subset name from filename like sweep_HumanEval.json."""
    match = re.search(r"sweep[s]?_(.+)\.json$", filepath.name)
    if match:
        return match.group(1)
    return filepath.stem


def parse_sweep_file(filepath: Path) -> list[dict]:
    """Parse a single sweep JSON and return rows for the CSV."""
    with open(filepath) as f:
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

        for metric_key, csv_key in METRICS_TO_EXTRACT:
            metric_data = metrics.get(metric_key, {})
            successful = metric_data.get("successful", {})
            row[csv_key] = successful.get("median", "")

        rows.append(row)

    return rows


def main() -> None:
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
    args = parser.parse_args()

    all_rows: list[dict] = []

    for filepath in args.files:
        if not filepath.exists():
            print(f"[WARN] File not found, skipping: {filepath}", file=sys.stderr)
            continue

        try:
            rows = parse_sweep_file(filepath)
            all_rows.extend(rows)
            print(f"[INFO] Parsed {len(rows)} entries from {filepath.name}")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to parse {filepath}: {e}", file=sys.stderr)
            continue

    if not all_rows:
        print("[ERROR] No data was successfully parsed", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[INFO] CSV written to: {args.output} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
