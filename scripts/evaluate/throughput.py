"""Throughput benchmark: all subsets mixed into one recorded, concurrent run.

A throughput run's deliverable is speculative-decoding *acceptance* per subset
(HumanEval, qa, ...). The legacy path ran each subset in a separate guidellm
pass and read acceptance off the server's global Prometheus counters, diffed
before/after -- which *forced* the subsets to run one at a time (a shared
counter can't be attributed to a subset while others run concurrently).

Here every request carries its subset in ``metadata`` and every response's raw
per-request metrics are recorded to the Parquet table, so all subsets can be
mixed into a *single* max-concurrency run and sliced apart afterward. One
saturated pass instead of N sequential ones -- and the same recording yields
per-subset acceptance plus an aggregate throughput figure.

This is a thin adapter, mirroring ``mrcr.py``: it only turns the benchmark
datasets into a deterministically-sampled, subset-tagged request set. The
generic runner (``request_runner``) sends them and records everything; the
analysis layer (``acceptance_report``) parses the recording. No inference,
persistence, or acceptance math lives here.

Server requirements: ``--per-request-spec-decode-metrics detailed`` so each
response carries its own ``speculative_decoding`` block. Unlike long-context,
the render endpoint is *not* needed -- prompts are short, so the fit-test is
skipped and prompt length comes from the response's usage block.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import TYPE_CHECKING

from acceptance_report import load_spec_records
from huggingface_hub import hf_hub_download
from perf_utils import CsvWriter
from request_runner import (
    TABLE_DIRNAME,
    Request,
    load_table,
    run_requests,
    stable_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("evaluate")

# Requested generation budget per request. Most responses stop at EOS well
# before this; it only caps the pathological long ones.
DEFAULT_MAX_NEW_TOKENS = 2048

# Requests per Parquet part file. One part per chunk gives crash-safe durability
# for long runs; large enough that the between-chunk barrier costs little of the
# saturated concurrency the run is trying to measure.
REQUESTS_PER_PART = 1000

# Chars of the prompt kept in metadata so the best/worst report can name a prompt
# without reloading the dataset. The full prompt is never stored in the table.
PROMPT_PREVIEW_CHARS = 160


def _load_subset_records(dataset: str, subset: str) -> list[dict]:
    """Load one subset's JSONL records from an HF dataset repo or a local dir."""
    from pathlib import Path  # noqa: PLC0415

    local = Path(dataset) / f"{subset}.jsonl"
    path = (
        local
        if local.exists()
        else Path(hf_hub_download(dataset, f"{subset}.jsonl", repo_type="dataset"))
    )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_request(
    subset: str, source_index: int, record: dict, max_new_tokens: int
) -> Request:
    """Build one request, tagging it with a stable pointer back to its source row.

    ``source_index`` is the record's position in ``<subset>.jsonl`` (not the
    sampled order), so any downstream report can recover the full prompt with a
    direct ``jsonl[source_index]`` lookup -- no need to replay the sampler. The
    ``prompt_sha`` lets that lookup self-verify against dataset drift.
    """
    prompt = str(record["prompt"])
    return Request(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        request_id=f"{subset}-{source_index}",
        metadata={
            "subset": subset,
            "source_index": source_index,
            "prompt_sha": stable_hash(prompt)[:16],
            "question_id": record.get("question_id") or record.get("task_id"),
            "category": record.get("category"),
            "prompt_preview": " ".join(prompt[:PROMPT_PREVIEW_CHARS].split()),
        },
    )


def _select_mixed(
    dataset: str,
    subsets: list[str],
    *,
    samples_per_subset: int,
    max_new_tokens: int,
    selection_seed: int,
) -> list[Request]:
    """Deterministically sample per subset, then mix all subsets together.

    Within a subset, records are ranked by a stable content hash and the lowest
    ``samples_per_subset`` are kept -- reproducible and independent of load
    order. Each kept request keeps its *source* row index (its position in
    ``<subset>.jsonl``, not the sampled order), so a report can recover the full
    prompt by a direct lookup. The kept requests from every subset are then
    interleaved by a stable hash of their id so the concurrent run exercises all
    subsets at once rather than subset-by-subset.
    """
    selected: list[Request] = []
    for subset in subsets:
        records = _load_subset_records(dataset, subset)
        ranked = sorted(
            enumerate(records),
            key=lambda ir: stable_hash(f"{selection_seed}:{subset}:{ir[1]['prompt']}"),
        )
        kept = ranked[:samples_per_subset]
        selected.extend(
            _to_request(subset, src, rec, max_new_tokens) for src, rec in kept
        )
        logger.info("  %s: %d/%d records", subset, len(kept), len(records))
    selected.sort(key=lambda req: stable_hash(req.request_id))
    return selected


def _write_subset_report(output_dir: Path, records: list) -> None:
    """Aggregate per-request acceptance by subset and write a CSV + table."""
    buckets: dict[str, dict] = {}
    for r in records:
        subset = r.metadata.get("subset", "?")
        b = buckets.setdefault(
            subset,
            {
                "subset": subset,
                "num_requests": 0,
                "num_spec_steps": 0,
                "num_draft_tokens": 0,
                "num_accepted_draft_tokens": 0,
            },
        )
        b["num_requests"] += 1
        b["num_spec_steps"] += r.spec["num_spec_steps"]
        b["num_draft_tokens"] += r.spec["num_draft_tokens"]
        b["num_accepted_draft_tokens"] += r.spec["num_accepted_draft_tokens"]

    rows = []
    for b in sorted(buckets.values(), key=lambda x: x["subset"]):
        b["draft_acceptance_rate"] = (
            b["num_accepted_draft_tokens"] / b["num_draft_tokens"]
            if b["num_draft_tokens"]
            else 0.0
        )
        b["mean_acceptance_length"] = (
            1 + b["num_accepted_draft_tokens"] / b["num_spec_steps"]
            if b["num_spec_steps"]
            else 0.0
        )
        rows.append(b)

    print("\n=== Acceptance by subset ===\n")
    print(f"  {'Subset':<18} {'Requests':>9} {'Accept Rate':>12} {'Mean Len':>9}")
    print("  " + "-" * 51)
    for row in rows:
        print(
            f"  {row['subset']:<18} {row['num_requests']:>9} "
            f"{row['draft_acceptance_rate']:>12.4f} "
            f"{row['mean_acceptance_length']:>9.4f}"
        )
    print()

    CsvWriter(
        output_dir / "acceptance_by_subset.csv",
        [
            "subset",
            "num_requests",
            "num_spec_steps",
            "num_draft_tokens",
            "num_accepted_draft_tokens",
            "draft_acceptance_rate",
            "mean_acceptance_length",
        ],
    ).append_rows(rows)


def _write_throughput_summary(
    output_dir: Path, table_dir: Path, wall_time_s: float
) -> None:
    """Aggregate wall-clock throughput across the whole mixed run."""
    table = load_table(table_dir)
    ok = [r for r in table.to_pylist() if r["status"] == "ok"]
    completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    summary = {
        "num_ok_requests": len(ok),
        "completion_tokens": completion_tokens,
        "wall_time_s": round(wall_time_s, 2),
        "output_tokens_per_s": round(completion_tokens / wall_time_s, 1)
        if wall_time_s
        else 0.0,
        "requests_per_s": round(len(ok) / wall_time_s, 3) if wall_time_s else 0.0,
    }
    with (output_dir / "throughput_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Aggregate throughput: %d reqs, %d output tokens in %.1fs "
        "(%.1f tok/s, %.2f req/s)",
        summary["num_ok_requests"],
        summary["completion_tokens"],
        summary["wall_time_s"],
        summary["output_tokens_per_s"],
        summary["requests_per_s"],
    )


def _request_acceptance(record) -> float:
    """Per-request draft acceptance rate (accepted / drafted)."""
    drafted = record.spec["num_draft_tokens"]
    return record.spec["num_accepted_draft_tokens"] / drafted if drafted else 0.0


def _print_extremes(records: list, top_n: int) -> None:
    """Print the highest- and lowest-acceptance prompts within each subset."""
    if top_n <= 0:
        return
    by_subset: dict[str, list] = {}
    for r in records:
        by_subset.setdefault(r.metadata.get("subset", "?"), []).append(r)

    print(f"\n=== Best/worst {top_n} prompts per subset (by acceptance rate) ===")
    for subset in sorted(by_subset):
        ranked = sorted(by_subset[subset], key=_request_acceptance, reverse=True)
        n = len(ranked)
        # Best from the front, worst from the back; never show a request twice
        # when the subset has fewer than 2*top_n requests.
        best_idx = list(range(min(top_n, n)))
        worst_idx = list(range(n - 1, max(n - top_n, top_n) - 1, -1))
        print(f"\n  {subset}:")
        for tag, idxs in (("best", best_idx), ("worst", worst_idx)):
            for i in idxs:
                r = ranked[i]
                preview = r.metadata.get("prompt_preview", "")
                print(
                    f"    [{tag:>5}] rate={_request_acceptance(r):.4f} "
                    f"len={r.spec['mean_acceptance_length']:.3f}  {preview}"
                )


def run_throughput(
    target: str,
    model_info: dict | None,
    dataset: str,
    subsets: list[str],
    output_dir: Path,
    *,
    max_concurrency: int,
    samples_per_subset: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    selection_seed: int = 0,
    top_n_prompts: int = 0,
) -> None:
    """Run all subsets in one mixed, concurrent pass and report acceptance."""
    if model_info is None:
        logger.error("Could not determine served model info from %s/models", target)
        sys.exit(1)
    model_name = model_info["id"]
    root_url = target.rstrip("/").removesuffix("/v1")

    logger.info("Loading %d subsets from %s", len(subsets), dataset)
    selected = _select_mixed(
        dataset,
        subsets,
        samples_per_subset=samples_per_subset,
        max_new_tokens=max_new_tokens,
        selection_seed=selection_seed,
    )
    if not selected:
        logger.error("No requests selected from dataset %s", dataset)
        sys.exit(1)
    logger.info(
        "Selected %d requests across %d subsets; running mixed at concurrency %d",
        len(selected),
        len(subsets),
        max_concurrency,
    )

    batches = [
        selected[i : i + REQUESTS_PER_PART]
        for i in range(0, len(selected), REQUESTS_PER_PART)
    ]
    start = time.perf_counter()
    table_dir = run_requests(
        root_url,
        model_name,
        batches,
        output_dir / TABLE_DIRNAME,
        max_concurrency=max_concurrency,
        fit_test=False,  # short prompts: skip render, use usage for prompt length
    )
    wall_time_s = time.perf_counter() - start

    records = load_spec_records(table_dir)
    if not records:
        logger.error(
            "No spec-decode records recorded. Was the server started with "
            "--per-request-spec-decode-metrics detailed?"
        )
        sys.exit(1)
    _write_subset_report(output_dir, records)
    _print_extremes(records, top_n_prompts)
    _write_throughput_summary(output_dir, table_dir, wall_time_s)
