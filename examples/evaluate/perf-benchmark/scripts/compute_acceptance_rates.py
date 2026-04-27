#!/usr/bin/env python3
"""Compute per-subset speculative-decoding acceptance rates from before/after snapshots.

Each snapshot is a JSON written by get_acceptance_rate.py.  We subtract the
before-snapshot counters from the after-snapshot counters so the result
reflects only the traffic generated during that subset's sweep, not the
cumulative server lifetime.

Per-position rates are computed from raw per_position_counts (integer counters)
rather than the stored rates, so the delta is mathematically correct.

Usage:
    python compute_acceptance_rates.py \\
        --acceptance-dir perf_results/acceptance \\
        --subsets HumanEval,math_reasoning,qa \\
        --output perf_results/acceptance_rates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_delta(before: dict, after: dict) -> dict:
    """Return acceptance metrics for the traffic between two snapshots."""
    drafts = after["num_drafts"] - before["num_drafts"]
    draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]

    acceptance_rate = accepted / draft_tokens if draft_tokens > 0 else 0.0
    mean_accepted = 1 + accepted / drafts if drafts > 0 else 0.0

    # Per-position: use raw counts so subtraction is valid.
    before_counts: list[int] = before.get("per_position_counts", [])
    after_counts: list[int] = after.get("per_position_counts", [])
    n = max(len(before_counts), len(after_counts))
    before_counts += [0] * (n - len(before_counts))
    after_counts += [0] * (n - len(after_counts))
    delta_counts = [a - b for a, b in zip(after_counts, before_counts)]
    per_position_acceptance = (
        [round(c / drafts, 4) for c in delta_counts] if drafts > 0 else [0.0] * n
    )

    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "acceptance_rate": round(acceptance_rate, 4),
        "mean_accepted_tokens": round(mean_accepted, 4),
        "per_position_acceptance": per_position_acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-subset acceptance rates from before/after snapshots."
    )
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        required=True,
        help="Directory containing before_SUBSET.json / after_SUBSET.json files.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        required=True,
        help="Comma-separated list of subset names.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON file for per-subset acceptance rates.",
    )
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",")]
    acc_dir = args.acceptance_dir
    result: dict[str, dict] = {}

    for subset in subsets:
        before_path = acc_dir / f"before_{subset}.json"
        after_path = acc_dir / f"after_{subset}.json"

        if not after_path.exists():
            print(
                f"[WARN] Missing after snapshot for {subset} — skipping.",
                file=sys.stderr,
            )
            continue

        after = load_snapshot(after_path)

        if before_path.exists():
            # Delta mode: server was shared across subsets; subtract baseline.
            before = load_snapshot(before_path)
            delta = compute_delta(before, after)
            mode = "delta"
        else:
            # Fresh-server mode: server restarted before this subset so counters
            # started at zero — the after snapshot IS the per-subset rate.
            n_pos = len(after.get("per_position_counts", []))
            drafts = after["num_drafts"]
            delta = {
                "num_drafts": drafts,
                "num_draft_tokens": after["num_draft_tokens"],
                "num_accepted_tokens": after["num_accepted_tokens"],
                "acceptance_rate": after["acceptance_rate"],
                "mean_accepted_tokens": after["mean_accepted_tokens"],
                "per_position_acceptance": (
                    [round(c / drafts, 4) for c in after["per_position_counts"]]
                    if drafts > 0 and n_pos > 0
                    else after.get("per_position_acceptance", [])
                ),
            }
            mode = "fresh-server"

        result[subset] = delta

        ar = delta["acceptance_rate"]
        mean = delta["mean_accepted_tokens"]
        per_pos = "  ".join(f"{v:.1%}" for v in delta["per_position_acceptance"])
        print(f"  {subset:<22} acceptance={ar:.1%}  mean_accepted={mean:.2f}  [{mode}]")
        print(f"    per-position: {per_pos}")

    if not result:
        print("[ERROR] No subsets had complete before/after snapshots.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n[INFO] Acceptance rates written to: {args.output}")


if __name__ == "__main__":
    main()
