"""Run short acceptance evaluations and aggregate per-request counters."""

from __future__ import annotations

import argparse  # noqa: TC003
import asyncio
import csv
import json
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path  # noqa: TC003

from request_runner import collect_requests

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


def acceptance_summary(results: list[dict]) -> dict:
    """Aggregate response counters before dividing into rates."""
    successful = [r for r in results if r["status"] == "ok"]
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
        "requests": len(results),
        "successful_requests": len(successful),
        "errored_requests": len(results) - len(successful),
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
        if result["status"] == "ok":
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
    """Use source-provided categories and include failures in their group counts."""
    groups = defaultdict(list)
    for result in results:
        if category := result.get("category"):
            groups[category].append(result)
    return {name: acceptance_summary(rows) for name, rows in sorted(groups.items())}


def check_results(summary: dict) -> bool:
    """Require successful, error-free requests with speculative metrics."""
    if not summary["successful_requests"] or summary["errored_requests"]:
        return False
    if summary["requests_without_spec_metrics"]:
        print(
            "Missing per-request acceptance metrics. Start vLLM with "
            "--per-request-spec-decode-metrics summary and use n=1.",
            file=sys.stderr,
        )
        return False
    return True


def _csv_row(
    name: str, summary: dict, elapsed: float | None, *, group: str = "dataset"
) -> dict:
    return {
        group: name,
        **{k: v for k, v in summary.items() if k != "acceptance_rate_by_position"},
        **{
            f"acceptance_rate_pos_{position}": rate
            for position, rate in summary["acceptance_rate_by_position"].items()
        },
        **({"elapsed_seconds": elapsed} if elapsed is not None else {}),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = dict.fromkeys(key for row in rows for key in row)
    if "elapsed_seconds" in fields:
        fields.pop("elapsed_seconds")
        fields["elapsed_seconds"] = None
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace, requests: list[dict]) -> int:
    """Finish each dataset and save its results before starting the next."""
    limit = args.max_requests if args.max_requests is not None else len(requests)
    cycles, remaining = divmod(limit, len(requests))
    results = []
    report: dict = {
        "schema_version": 1,
        "selected_datasets": list(dict.fromkeys(r["dataset"] for r in requests)),
        "source_selection": getattr(args, "dataset_selection", {}),
        "overall": acceptance_summary([]),
        "by_dataset": {},
        "by_prompt_length": {},
        "by_category": {},
    }
    start = time.perf_counter()
    csv_rows = []
    csv_path = args.output_dir / "summary.csv"
    if limit < len(requests):
        print(
            "Warning: --max-requests is a total ordered limit; overall describes "
            "only requests actually sent. Use --max-samples for per-dataset coverage.",
            file=sys.stderr,
        )
    with (args.output_dir / "requests.jsonl").open("x") as raw:
        for dataset, group in groupby(requests, key=itemgetter("dataset")):
            samples = list(group)
            count = cycles * len(samples) + min(remaining, len(samples))
            remaining = max(0, remaining - len(samples))
            if not count:
                continue
            print(f"Collecting {dataset}: {count} requests", flush=True)
            dataset_start = time.perf_counter()
            collected = asyncio.run(
                collect_requests(
                    samples,
                    args.target,
                    args.concurrency,
                    raw,
                    args.timeout,
                    count=count,
                )
            )
            elapsed = time.perf_counter() - dataset_start
            results.extend(collected)
            summary = acceptance_summary(collected)
            report["by_dataset"][dataset] = summary
            report["overall"] = acceptance_summary(results)
            report["by_prompt_length"] = acceptance_by_prompt_length(results)
            report["by_category"] = acceptance_by_category(results)
            for group_name in ("prompt_length", "category"):
                summaries = report[f"by_{group_name}"]
                if summaries:
                    _write_csv(
                        args.output_dir / f"summary_by_{group_name}.csv",
                        [
                            _csv_row(name, stats, None, group=group_name)
                            for name, stats in summaries.items()
                        ],
                    )
            (args.output_dir / "summary.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            csv_rows.append(_csv_row(dataset, summary, elapsed))
            _write_csv(csv_path, csv_rows)
            print(
                f"{dataset}: acceptance_rate={summary['acceptance_rate']}, "
                f"acceptance_length={summary['acceptance_length']} ({elapsed:.1f}s)",
                flush=True,
            )
        csv_rows.append(
            _csv_row("overall", report["overall"], time.perf_counter() - start)
        )
        _write_csv(csv_path, csv_rows)
    skipped = set(report["selected_datasets"]) - report["by_dataset"].keys()
    if skipped:
        print(
            f"Warning: datasets not visited: {', '.join(sorted(skipped))}",
            file=sys.stderr,
        )
    print(
        f"Finished: {report['overall']['successful_requests']} successful, "
        f"{report['overall']['errored_requests']} errors, "
        f"{report['overall']['requests_without_spec_metrics']} missing metrics",
        flush=True,
    )
    return 0 if check_results(report["overall"]) else 1
