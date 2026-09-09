"""MRCR long-context benchmark: a set of requests + an ordering + a stop rule.

This is a thin adapter. It knows only how to turn OpenAI's ``openai/mrcr``
dataset into a deterministically-selected, length-ordered set of chat requests;
the generic runner (``request_runner``) sends them and records everything, and
the analysis layer (``acceptance_report``) turns the recording into reports.
Nothing about inference, persistence, or binning lives here.

Deterministic superset selection
---------------------------------
Records are stratified into MRCR's native power-of-2 token bins and, within each
bin, ranked by a stable content hash; the lowest-ranked ``samples_per_bin`` per
bin are kept. Bin assignment uses ``n_chars / EST_CHARS_PER_TOKEN`` (MRCR carries
no token count, and this ratio is ~4.9 with <=2% spread across the whole dataset
-- validated offline against o200k -- so it matches exact tokenization for bin
assignment at zero runtime cost). The candidate set is independent of the
server's ``max_model_len``; the only context-dependent step is the render
fit-test in the runner, which is monotonic, so a larger-context run selects a
superset of a smaller one. Batches are handed to the runner smallest-bin-first
with a stop rule that halts once an entire bin renders oversized.

Correctness is intentionally not graded: MRCR is used purely as a source of
long, multi-turn prompts to study how acceptance holds up as context grows.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd
from acceptance_report import (
    DEFAULT_CONTEXT_BIN_EDGES,
    DEFAULT_POSITION_BIN_SIZE,
    load_spec_records,
    write_report,
)
from huggingface_hub import hf_hub_download
from request_runner import (
    TABLE_DIRNAME,
    Request,
    run_requests,
    select_stratified,
    stable_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("evaluate")

MRCR_REPO = "openai/mrcr"
MRCR_SHARDS = ("2needle/2needle_0.parquet", "2needle/2needle_1.parquet")

# Chars-per-token ratio for MRCR content, used only to estimate a record's native
# token bin (exact length comes from the render endpoint). 4.9 +/- 0.1 across the
# full 20k-490k token range measured with OpenAI's o200k tokenizer.
EST_CHARS_PER_TOKEN = 4.9

# Native token bins for selection: MRCR's power-of-2 edges minus the leading 0,
# since every record is far larger than 4096 tokens.
SELECTION_BIN_EDGES = tuple(e for e in DEFAULT_CONTEXT_BIN_EDGES if e > 0)

# Requested generation budget; the render endpoint clips this to whatever fits.
MAX_NEW_TOKENS = 512
# Drop candidates left with less than this much room -- too few steps to measure.
MIN_GENERATION_ROOM = 32


def _load_records() -> list[dict]:
    frames = [
        pd.read_parquet(hf_hub_download(MRCR_REPO, shard, repo_type="dataset"))
        for shard in MRCR_SHARDS
    ]
    return pd.concat(frames, ignore_index=True).to_dict("records")


def _to_request(index: int, record: dict) -> Request:
    return Request(
        messages=json.loads(record["prompt"]),
        max_tokens=MAX_NEW_TOKENS,
        request_id=f"mrcr-{index}",
        metadata={
            "index": index,
            "n_chars": int(record["n_chars"]),
            "est_tokens": round(record["n_chars"] / EST_CHARS_PER_TOKEN),
            "n_needles": int(record["n_needles"]),
            "total_messages": int(record["total_messages"]),
        },
    )


def _all_oversized(rows: list[dict]) -> bool:
    """Stop rule: an entire bin rendered oversized (all larger bins are bigger).

    Keyed on real render outcomes, never the length estimate. A transport error
    leaves fit unknown, so it does not count as oversized and won't stop the run.
    """
    return bool(rows) and all(r["status"] == "oversized" for r in rows)


def run_mrcr(
    target: str,
    model_info: dict | None,
    output_dir: Path,
    max_concurrency: int,
    *,
    samples_per_bin: int,
    selection_seed: int = 0,
    position_bin_size: int = DEFAULT_POSITION_BIN_SIZE,
) -> None:
    """Select MRCR records, run them through the generic harness, and report."""
    if model_info is None:
        logger.error("Could not determine served model info from %s/models", target)
        sys.exit(1)
    model_name = model_info["id"]
    logger.info(
        "Server reports model=%s max_model_len=%s",
        model_name,
        model_info.get("max_model_len"),
    )
    root_url = target.rstrip("/").removesuffix("/v1")

    logger.info("Loading MRCR 2-needle dataset...")
    records = _load_records()
    if not records:
        logger.error("MRCR dataset returned no records")
        sys.exit(1)
    logger.info("Loaded %d MRCR records", len(records))

    indexed = list(enumerate(records))
    selected = select_stratified(
        indexed,
        bin_edges=SELECTION_BIN_EDGES,
        samples_per_bin=samples_per_bin,
        key_fn=lambda ir: stable_hash(f"{selection_seed}:{ir[1]['prompt']}"),
        length_fn=lambda ir: ir[1]["n_chars"] / EST_CHARS_PER_TOKEN,
    )
    batches = [[_to_request(i, rec) for i, rec in group] for group in selected]
    n_selected = sum(len(b) for b in batches)
    logger.info(
        "Selected %d candidate records across %d bins (%d per bin)",
        n_selected,
        len(batches),
        samples_per_bin,
    )

    table_dir = run_requests(
        root_url,
        model_name,
        batches,
        output_dir / TABLE_DIRNAME,
        max_concurrency=max_concurrency,
        fit_test=True,
        min_generation_room=MIN_GENERATION_ROOM,
        should_stop=_all_oversized,
    )

    records_out = load_spec_records(table_dir)
    if not records_out:
        logger.error("No usable spec-decode results were recorded")
        sys.exit(1)
    write_report(
        output_dir,
        records_out,
        context_bin_edges=DEFAULT_CONTEXT_BIN_EDGES,
        position_bin_size=position_bin_size,
    )
    logger.info(
        "Render the acceptance figure with:\n"
        "  python plot.py acceptance --table %s --output %s",
        table_dir,
        output_dir / "acceptance.png",
    )
