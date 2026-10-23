"""Run short acceptance evaluations and aggregate per-request counters."""

from __future__ import annotations

import argparse  # noqa: TC003
import asyncio
import csv
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path  # noqa: TC003

from request_runner import MAX_CONSECUTIVE_FAILURES, collect_requests

PROMPT_TOKEN_BUCKET_BOUNDARIES = (
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)


def acceptance_summary(successful: list[dict]) -> dict:
    """Aggregate successful response counters before dividing into rates."""
    specs = [
        spec
        for result in successful
        if (spec := (result.get("metrics") or {}).get("speculative_decoding"))
        is not None
    ]
    steps = sum(s["num_spec_steps"] for s in specs)
    accepted = sum(s["num_accepted_draft_tokens"] for s in specs)
    drafted = sum(s["num_draft_tokens"] for s in specs)
    histograms = [s["acceptance_histogram"] for s in specs]
    histogram = [sum(counts) for counts in zip(*histograms, strict=True)]
    by_position = {}
    remaining = steps - histogram[0] if histogram else 0
    for position in range(1, len(histogram)):
        by_position[str(position)] = remaining / steps if steps else None
        remaining -= histogram[position]
    return {
        "requests": len(successful),
        "successful_requests": len(successful),
        "requests_with_spec_metrics": len(specs),
        "requests_without_spec_metrics": len(successful) - len(specs),
        "requests_with_zero_drafts": sum(s["num_spec_steps"] == 0 for s in specs),
        "num_spec_steps": steps,
        "num_accepted_draft_tokens": accepted,
        "num_draft_tokens": drafted,
        "acceptance_length": 1 + accepted / steps if steps else None,
        "acceptance_rate": accepted / drafted if drafted else None,
        "acceptance_rate_by_position": by_position,
    }


def acceptance_by_prompt_length(results: list[dict]) -> dict:
    """Group successful requests by the prompt token counts returned by vLLM."""
    groups = defaultdict(list)
    for result in results:
        index = bisect_left(PROMPT_TOKEN_BUCKET_BOUNDARIES, result["prompt_tokens"])
        groups[index].append(result)
    summaries = {}
    for index, rows in sorted(groups.items()):
        low = PROMPT_TOKEN_BUCKET_BOUNDARIES[index - 1] + 1 if index else 0
        label = (
            f"{low}-{PROMPT_TOKEN_BUCKET_BOUNDARIES[index]}"
            if index < len(PROMPT_TOKEN_BUCKET_BOUNDARIES)
            else f">{PROMPT_TOKEN_BUCKET_BOUNDARIES[-1]}"
        )
        summaries[label] = acceptance_summary(rows)
    return summaries


def acceptance_by_category(results: list[dict]) -> dict:
    """Group successful requests by source-provided categories."""
    groups = defaultdict(list)
    for result in results:
        if category := result.get("category"):
            groups[category].append(result)
    return {name: acceptance_summary(rows) for name, rows in sorted(groups.items())}


def check_results(summary: dict) -> bool:
    """Require successful requests with speculative metrics."""
    if not summary["successful_requests"]:
        return False
    if summary["requests_without_spec_metrics"]:
        print(
            "Missing per-request acceptance metrics. Start vLLM with "
            "--per-request-spec-decode-metrics summary and use n=1.",
            file=sys.stderr,
        )
        return False
    return True


def _csv_row(name: str, summary: dict, *, group: str = "dataset") -> dict:
    return {
        group: name,
        **{k: v for k, v in summary.items() if k != "acceptance_rate_by_position"},
        **{
            f"acceptance_rate_pos_{position}": rate
            for position, rate in summary["acceptance_rate_by_position"].items()
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = dict.fromkeys(key for row in rows for key in row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace, requests: list[dict]) -> int:
    """Checkpoint datasets as they finish; aggregate cumulative results once."""
    all_successful = []
    total_failures = 0
    report: dict = {"schema_version": 1, "by_dataset": {}}
    csv_rows = []
    csv_path = args.output_dir / "summary.csv"
    consecutive_failures = 0
    with (args.output_dir / "requests.jsonl").open("x") as raw:
        try:
            for dataset, group in groupby(requests, key=itemgetter("dataset")):
                samples = list(group)
                print(f"Collecting {dataset}: {len(samples)} requests", flush=True)
                collected, consecutive_failures = asyncio.run(
                    collect_requests(
                        samples,
                        args.target,
                        args.concurrency,
                        raw,
                        args.timeout,
                        consecutive_failures,
                    )
                )
                successful = [r for r in collected if r["status"] == "ok"]
                all_successful.extend(successful)
                total_failures += len(collected) - len(successful)
                summary = acceptance_summary(successful)
                report["by_dataset"][dataset] = summary
                csv_rows.append(_csv_row(dataset, summary))
                _write_csv(csv_path, csv_rows)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        "Stopping after three consecutive request failures",
                        file=sys.stderr,
                    )
                    break
        finally:
            report["overall"] = acceptance_summary(all_successful)
            report["by_prompt_length"] = acceptance_by_prompt_length(all_successful)
            report["by_category"] = acceptance_by_category(all_successful)
            for group_name in ("prompt_length", "category"):
                summaries = report[f"by_{group_name}"]
                if summaries:
                    _write_csv(
                        args.output_dir / f"summary_by_{group_name}.csv",
                        [
                            _csv_row(name, stats, group=group_name)
                            for name, stats in summaries.items()
                        ],
                    )
            (args.output_dir / "summary.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        csv_rows.append(_csv_row("overall", report["overall"]))
        _write_csv(csv_path, csv_rows)
    print(f"Failed requests: {total_failures}", file=sys.stderr)
    return 0 if not total_failures and check_results(report["overall"]) else 1
