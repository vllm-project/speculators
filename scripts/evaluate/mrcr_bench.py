"""Long-context acceptance benchmarking via OpenAI's MRCR dataset.

Thin dataset adapter over the generic per-request spec-decode acceptance
harness in ``spec_acceptance``. This module knows only how to load the
``openai/mrcr`` 2-needle dataset and turn a record into chat messages / a
selection key / a length proxy; the harness does the rendering, generation,
persistence, binning and reporting.

Sample selection is deterministic and monotonic in ``max_model_len`` (see
``spec_acceptance.select_stratified``): records are stratified into MRCR's
native power-of-2 token bins and, within each bin, the lowest-ranked
``samples_per_bin`` by a stable content hash are chosen. The bin a record lands
in is estimated from its ``n_chars`` metadata -- MRCR carries no per-record
token count, but ``n_chars / tokens`` is empirically ~4.9 with <=2% spread
across the whole dataset (validated offline against OpenAI's o200k tokenizer),
so the estimate matches exact tokenization for bin assignment at zero runtime
cost. Exact prompt length and the context fit-test always come from the
server's render endpoint, never the estimate.

Correctness is intentionally not graded: MRCR is used here purely as a source
of long, multi-turn prompts to see how spec-decode acceptance holds up as
context length grows and generation proceeds, not to measure retrieval accuracy.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd
from huggingface_hub import hf_hub_download
from spec_acceptance import (
    DEFAULT_CONTEXT_BIN_EDGES,
    DEFAULT_POSITION_BIN_SIZE,
    run_acceptance,
    select_stratified,
    stable_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("evaluate")

MRCR_REPO = "openai/mrcr"
MRCR_SHARDS = ("2needle/2needle_0.parquet", "2needle/2needle_1.parquet")

# Empirical chars-per-token ratio for MRCR content, used only to estimate which
# native token bin a record falls in (exact length comes from the render
# endpoint). Measured at 4.9 +/- 0.1 across the full 20k-490k token range with
# OpenAI's o200k tokenizer, so bin assignment matches exact tokenization.
EST_CHARS_PER_TOKEN = 4.9

# MRCR's native context-length bins (tokens); shared with the harness so
# selection and the report use one bin definition. Drop the leading 0 edge for
# selection -- every record is far larger than 4096 tokens.
SELECTION_BIN_EDGES = tuple(e for e in DEFAULT_CONTEXT_BIN_EDGES if e > 0)


def _iter_records():
    for shard in MRCR_SHARDS:
        path = hf_hub_download(MRCR_REPO, shard, repo_type="dataset")
        yield from pd.read_parquet(path).to_dict("records")


def _record_messages(record: dict) -> list[dict]:
    return json.loads(record["prompt"])


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
    """Run the MRCR long-context evaluation.

    Deterministically selects up to *samples_per_bin* records per native token
    bin, then renders + generates them against *target* and reports acceptance
    binned by context length two ways (see module docstring).
    """
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
    records = list(_iter_records())
    if not records:
        logger.error("MRCR dataset returned no records")
        sys.exit(1)
    logger.info("Loaded %d MRCR records", len(records))

    candidate_bins = select_stratified(
        records,
        bin_edges=SELECTION_BIN_EDGES,
        samples_per_bin=samples_per_bin,
        key_fn=lambda r: stable_hash(f"{selection_seed}:{r['prompt']}"),
        length_fn=lambda r: r["n_chars"] / EST_CHARS_PER_TOKEN,
    )
    n_selected = sum(len(b) for b in candidate_bins)
    logger.info(
        "Selected %d candidate records across %d bins (%d per bin)",
        n_selected,
        len(candidate_bins),
        samples_per_bin,
    )

    run_acceptance(
        root_url,
        model_name,
        candidate_bins,
        _record_messages,
        output_dir,
        max_concurrency=max_concurrency,
        context_bin_edges=DEFAULT_CONTEXT_BIN_EDGES,
        position_bin_size=position_bin_size,
    )
