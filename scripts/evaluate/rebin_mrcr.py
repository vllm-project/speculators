#!/usr/bin/env python3
"""Rebuild long-context acceptance reports from a raw results log.

`evaluate.py long-context` streams every raw per-request result (prompt length +
full spec-decode metrics) to ``raw_requests.jsonl`` in its output directory as
it runs. That file is the source of truth -- the aggregated CSVs are just one
slicing of it. This script re-slices it with different bucket edges, so changing
how results are binned doesn't require re-running inference against the server.

Usage:
    python rebin_mrcr.py --raw-log <output_dir>/raw_requests.jsonl
    python rebin_mrcr.py --raw-log <dir>/raw_requests.jsonl --position-bin-size 512
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spec_acceptance import (
    DEFAULT_CONTEXT_BIN_EDGES,
    DEFAULT_POSITION_BIN_SIZE,
    load_results,
    write_report,
)

logger = logging.getLogger("evaluate")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-log",
        type=Path,
        required=True,
        help="Path to a raw_requests.jsonl produced by `evaluate.py long-context`",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the re-bucketed CSVs (default: alongside --raw-log)",
    )
    parser.add_argument(
        "--context-bin-edges",
        type=str,
        default=None,
        help=(
            "Comma-separated token edges for the by-start-length report "
            f"(default: {','.join(map(str, DEFAULT_CONTEXT_BIN_EDGES))})"
        ),
    )
    parser.add_argument(
        "--position-bin-size",
        type=int,
        default=DEFAULT_POSITION_BIN_SIZE,
        help=(
            "Token-position bin width for the by-position report "
            f"(default: {DEFAULT_POSITION_BIN_SIZE})"
        ),
    )
    args = parser.parse_args()

    results = load_results(args.raw_log)
    if not results:
        logger.error("No results found in %s", args.raw_log)
        sys.exit(1)
    logger.info("Loaded %d raw results from %s", len(results), args.raw_log)

    context_bin_edges = DEFAULT_CONTEXT_BIN_EDGES
    if args.context_bin_edges:
        context_bin_edges = tuple(
            int(e) for e in args.context_bin_edges.split(",") if e.strip()
        )

    write_report(
        args.output_dir or args.raw_log.parent,
        results,
        context_bin_edges=context_bin_edges,
        position_bin_size=args.position_bin_size,
    )


if __name__ == "__main__":
    main()
