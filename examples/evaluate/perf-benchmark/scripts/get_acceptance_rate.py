#!/usr/bin/env python3
"""Query vLLM's Prometheus /metrics endpoint for speculative decoding acceptance rate.

Two usage modes:

  Snapshot (default): query once and report cumulative metrics.
    python get_acceptance_rate.py --endpoint http://localhost:8000

  Delta: snapshot before and after a shell command, report only the delta.
    python get_acceptance_rate.py --endpoint http://localhost:8000 \
        --run "guidellm benchmark --target http://localhost:8000/v1 ..."

Output is JSON with acceptance_rate, mean_accepted_tokens, per-position rates,
and raw counters.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

SPEC_DECODE_METRICS = {
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_emitted_tokens_total",
}

SPEC_DECODE_PER_POS_PREFIX = "vllm:spec_decode_num_accepted_tokens_per_pos"


def fetch_metrics(endpoint: str) -> str:
    url = endpoint.rstrip("/") + "/metrics"
    if not url.startswith(("http://", "https://")):
        print(f"[ERROR] Invalid URL scheme: {url}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    try:
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            return resp.read().decode()
    except URLError as e:
        print(  # noqa: T201
            f"[ERROR] Cannot reach {url}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def _strip_labels(name_with_labels: str) -> str:
    """Strip Prometheus label suffix: 'metric{k=v,...}' -> 'metric'."""
    brace = name_with_labels.find("{")
    return name_with_labels[:brace] if brace != -1 else name_with_labels


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition format into a flat dict of metric values.

    Metric names may carry label suffixes (e.g.
    ``vllm:spec_decode_num_drafts_total{engine="0",...}``).  We match on the
    bare metric name and store with the full key so per-position labels are
    preserved for downstream parsing.
    """
    min_parts = 2
    result: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < min_parts:
            continue
        full_name = parts[0]
        bare_name = _strip_labels(full_name)
        try:
            value = float(parts[1])
        except ValueError:
            continue

        if bare_name in SPEC_DECODE_METRICS or bare_name.startswith(
            SPEC_DECODE_PER_POS_PREFIX
        ):
            result[full_name] = value

    return result


def _sum_by_bare_name(raw: dict[str, float], bare_name: str) -> float:
    """Sum values across all label variants of a metric."""
    return sum(v for k, v in raw.items() if _strip_labels(k) == bare_name)


def extract_acceptance(
    raw: dict[str, float],
) -> dict[str, float | list[float]]:
    """Convert raw Prometheus counters into human-readable acceptance metrics."""
    num_drafts = _sum_by_bare_name(raw, "vllm:spec_decode_num_drafts_total")
    num_draft_tokens = _sum_by_bare_name(raw, "vllm:spec_decode_num_draft_tokens_total")
    num_accepted = _sum_by_bare_name(raw, "vllm:spec_decode_num_accepted_tokens_total")
    num_emitted = _sum_by_bare_name(raw, "vllm:spec_decode_num_emitted_tokens_total")

    acceptance_rate = num_accepted / num_draft_tokens if num_draft_tokens > 0 else 0.0
    mean_accepted = 1 + (num_accepted / num_drafts) if num_drafts > 0 else 0.0

    per_pos: dict[int, float] = {}
    for key, val in raw.items():
        bare = _strip_labels(key)
        if bare == SPEC_DECODE_PER_POS_PREFIX + "_total":
            try:
                pos_str = key.split('position="')[1].split('"')[0]
                pos = int(pos_str)
                per_pos[pos] = per_pos.get(pos, 0.0) + val
            except (IndexError, ValueError):
                continue

    n_pos = max(per_pos.keys()) + 1 if per_pos else 0
    per_pos_counts: list[int] = [int(per_pos.get(i, 0.0)) for i in range(n_pos)]
    per_pos_rates: list[float] = []
    if per_pos and num_drafts > 0:
        for i in range(n_pos):
            per_pos_rates.append(per_pos.get(i, 0.0) / num_drafts)

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "num_emitted_tokens": num_emitted,
        "acceptance_rate": round(acceptance_rate, 4),
        "mean_accepted_tokens": round(mean_accepted, 4),
        "per_position_counts": per_pos_counts,
        "per_position_acceptance": [round(r, 4) for r in per_pos_rates],
    }


def delta_metrics(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    """Subtract before-snapshot from after-snapshot."""
    all_keys = set(before.keys()) | set(after.keys())
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query vLLM /metrics for speculative decoding acceptance rate."
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="vLLM server base URL (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Shell command to run between two metric snapshots (delta mode).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write JSON output to file instead of stdout.",
    )
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[: -len("/v1")]

    if args.run:
        before_raw = parse_prometheus(fetch_metrics(endpoint))
        print(f"[INFO] Running: {args.run}", file=sys.stderr)  # noqa: T201
        subprocess.run(args.run, shell=True, check=True)  # noqa: S602
        after_raw = parse_prometheus(fetch_metrics(endpoint))
        raw = delta_metrics(before_raw, after_raw)
    else:
        raw = parse_prometheus(fetch_metrics(endpoint))

    result = extract_acceptance(raw)

    output = json.dumps(result, indent=2)
    if args.output:
        with Path(args.output).open("w") as f:
            f.write(output + "\n")
        print(f"[INFO] Results written to {args.output}", file=sys.stderr)  # noqa: T201
    else:
        print(output)  # noqa: T201


if __name__ == "__main__":
    main()
