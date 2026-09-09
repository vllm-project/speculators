"""Speculative-decoding acceptance analysis over a recorded request table.

Pure functions over the generic Parquet table written by ``request_runner``.
This layer is where all the spec-decode-specific and analysis-specific knowledge
lives -- how to read per-step acceptance arrays out of the recorded metrics, and
what context-length buckets to slice by. Crucially, the bucket edges are
*parameters here*, not constants baked into the runner: the same recording can be
re-sliced any number of ways (see ``rebin.py``) without touching inference.

Two slicings are produced:

* by prompt length at request start (:func:`bin_by_start_length`)
* by running token position of each verify step (:func:`bin_by_position`) --
  prompt length plus everything committed by earlier steps in the same request.

Requests must have been generated against a server started with
``--per-request-spec-decode-metrics detailed`` so each response carries its own
``speculative_decoding`` block (including the per-verify-step arrays).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from perf_utils import CsvWriter
from request_runner import load_table

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("evaluate")

# Context-length bucket edges (tokens) for the by-start-length report. These are
# OpenAI MRCR's native power-of-2 bins; they only slice results for reporting.
DEFAULT_CONTEXT_BIN_EDGES = (
    0,
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

# Bin width (tokens) for the by-token-position report -- finer than the context
# edges since acceptance can shift noticeably within a single request.
DEFAULT_POSITION_BIN_SIZE = 256


@dataclass
class SpecRecord:
    """One generated request's spec-decode stats, projected from the table."""

    prompt_tokens: int
    spec: dict  # the response's speculative_decoding block


def load_spec_records(table_dir: Path) -> list[SpecRecord]:
    """Project the recorded table down to successfully-generated spec-decode rows."""
    table = load_table(table_dir)
    records = []
    for row in table.to_pylist():
        if row["status"] != "ok" or not row["metrics_json"]:
            continue
        spec = json.loads(row["metrics_json"]).get("speculative_decoding")
        if not spec:
            continue
        records.append(SpecRecord(row["prompt_tokens"], spec))
    return records


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def _bucket_label(n_tokens: int, edges: tuple[int, ...]) -> str:
    for lo, hi in zip(edges, edges[1:], strict=False):
        if n_tokens <= hi:
            return f"{lo}-{hi}"
    return f"{edges[-1]}+"


def _position_bucket_label(n_tokens: int, bin_size: int) -> str:
    lo = (n_tokens // bin_size) * bin_size
    return f"{lo}-{lo + bin_size}"


def _bucket_sort_key(item: tuple[str, dict]) -> int:
    return int(item[0].split("-")[0].rstrip("+"))


def _finish_bucket(bucket: dict) -> dict:
    bucket["draft_acceptance_rate"] = (
        bucket["num_accepted_draft_tokens"] / bucket["num_draft_tokens"]
        if bucket["num_draft_tokens"]
        else 0.0
    )
    bucket["mean_acceptance_length"] = (
        1 + bucket["num_accepted_draft_tokens"] / bucket["num_spec_steps"]
        if bucket["num_spec_steps"]
        else 0.0
    )
    return bucket


def bin_by_start_length(
    records: list[SpecRecord],
    edges: tuple[int, ...] = DEFAULT_CONTEXT_BIN_EDGES,
) -> list[dict]:
    """Aggregate whole-request acceptance by prompt length at request start."""
    buckets: dict[str, dict] = {}
    for r in records:
        label = _bucket_label(r.prompt_tokens, edges)
        b = buckets.setdefault(
            label,
            {
                "bucket": label,
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
    return [_finish_bucket(b) for _, b in sorted(buckets.items(), key=_bucket_sort_key)]


def bin_by_position(
    records: list[SpecRecord],
    bin_size: int = DEFAULT_POSITION_BIN_SIZE,
) -> list[dict]:
    """Aggregate per-verify-step acceptance by the running token position.

    Each step's position is the prompt length plus everything committed by
    earlier steps in that same request (accepted draft tokens plus the
    always-accepted bonus token).
    """
    buckets: dict[str, dict] = {}
    for r in records:
        accepted = r.spec.get("per_step_accepted")
        drafted = r.spec.get("per_step_drafted")
        if not accepted or not drafted:
            continue
        pos = r.prompt_tokens
        for a, d in zip(accepted, drafted, strict=True):
            label = _position_bucket_label(pos, bin_size)
            b = buckets.setdefault(
                label,
                {
                    "bucket": label,
                    "num_spec_steps": 0,
                    "num_draft_tokens": 0,
                    "num_accepted_draft_tokens": 0,
                },
            )
            b["num_spec_steps"] += 1
            b["num_draft_tokens"] += d
            b["num_accepted_draft_tokens"] += a
            pos += a + 1
    return [_finish_bucket(b) for _, b in sorted(buckets.items(), key=_bucket_sort_key)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_bucket_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===\n")
    has_requests = "num_requests" in rows[0]
    header = f"  {'Bucket':<18}"
    if has_requests:
        header += f" {'Requests':>9}"
    header += f" {'Steps':>8} {'Accept Rate':>12} {'Mean Len':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        line = f"  {row['bucket']:<18}"
        if has_requests:
            line += f" {row['num_requests']:>9}"
        line += (
            f" {row['num_spec_steps']:>8} {row['draft_acceptance_rate']:>12.4f}"
            f" {row['mean_acceptance_length']:>9.4f}"
        )
        print(line)
    print()


def write_report(
    output_dir: Path,
    records: list[SpecRecord],
    *,
    context_bin_edges: tuple[int, ...] = DEFAULT_CONTEXT_BIN_EDGES,
    position_bin_size: int = DEFAULT_POSITION_BIN_SIZE,
) -> None:
    """Print and write both acceptance CSVs from projected spec records."""
    if not records:
        logger.error("No spec-decode records to report on")
        return

    by_start = bin_by_start_length(records, context_bin_edges)
    _print_bucket_table("Acceptance by context length at request start", by_start)
    CsvWriter(
        output_dir / "acceptance_by_start_length.csv",
        [
            "bucket",
            "num_requests",
            "num_spec_steps",
            "num_draft_tokens",
            "num_accepted_draft_tokens",
            "draft_acceptance_rate",
            "mean_acceptance_length",
        ],
    ).append_rows(by_start)

    by_position = bin_by_position(records, position_bin_size)
    if not by_position:
        logger.warning(
            "No per-step data; skipping by-position report (was the server "
            "started with --per-request-spec-decode-metrics detailed?)"
        )
        return
    _print_bucket_table("Acceptance by context length at token position", by_position)
    CsvWriter(
        output_dir / "acceptance_by_position.csv",
        [
            "bucket",
            "num_spec_steps",
            "num_draft_tokens",
            "num_accepted_draft_tokens",
            "draft_acceptance_rate",
            "mean_acceptance_length",
        ],
    ).append_rows(by_position)
