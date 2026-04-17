#!/usr/bin/env python3
"""Parse guidellm gen-len output files and compute per-subset max_tokens.

For each input JSON, extracts output_tokens from successful requests,
computes the median, and derives max_tokens as the first power of 2
greater than or equal to the median.

Usage:
    python parse_gen_len.py --output max_tokens.json gen_len_*.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path


def extract_subset_name(filepath: Path) -> str:
    """Derive subset name from filename like gen_len_HumanEval.json.

    Strips any prefix up to and including 'gen_len_' to handle filenames
    such as 'gen_len_HumanEval.json' or 'guidellm_gen_len_HumanEval.json'.
    """
    match = re.search(r"gen_len_(.+)\.json$", filepath.name)
    if match:
        return match.group(1)
    return filepath.stem


def next_power_of_two(value: float) -> int:
    if value <= 0:
        return 1
    return 2 ** math.ceil(math.log2(value))


def parse_gen_len_file(filepath: Path) -> dict:
    """Parse a single gen-len JSON and return statistics."""
    with open(filepath) as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        raise ValueError(f"No benchmarks found in {filepath}")

    successful = benchmarks[0].get("requests", {}).get("successful", [])
    if not successful:
        raise ValueError(f"No successful requests found in {filepath}")

    output_tokens = [r["output_tokens"] for r in successful]
    median = statistics.median(output_tokens)
    max_tokens = next_power_of_two(median)

    return {
        "count": len(output_tokens),
        "median": median,
        "min": min(output_tokens),
        "max": max(output_tokens),
        "max_tokens": max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse gen-len outputs and compute max_tokens per subset."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Gen-len JSON files (e.g. gen_len_HumanEval.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON file for max_tokens mapping",
    )
    args = parser.parse_args()

    max_tokens_map: dict[str, int] = {}

    print(f"{'Subset':<20} {'Count':>6} {'Median':>8} {'Min':>8} {'Max':>8} {'max_tokens':>12}")
    print("-" * 70)

    for filepath in args.files:
        if not filepath.exists():
            print(f"[WARN] File not found, skipping: {filepath}", file=sys.stderr)
            continue

        subset = extract_subset_name(filepath)
        try:
            stats = parse_gen_len_file(filepath)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to parse {filepath}: {e}", file=sys.stderr)
            continue

        max_tokens_map[subset] = stats["max_tokens"]

        print(
            f"{subset:<20} {stats['count']:>6} {stats['median']:>8.0f} "
            f"{stats['min']:>8} {stats['max']:>8} {stats['max_tokens']:>12}"
        )

    if not max_tokens_map:
        print("[ERROR] No files were successfully parsed", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(max_tokens_map, f, indent=2)

    print(f"\nmax_tokens mapping written to: {args.output}")


if __name__ == "__main__":
    main()
