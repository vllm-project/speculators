#!/usr/bin/env python3
"""Rebuild acceptance reports from a recorded request table, with no server.

The long-context runner streams every request's raw result (identity, exact
prompt length, response, and full server metrics) to a Parquet table -- the
source of truth. The aggregated CSVs are just one slicing of it. This script
re-slices that table with different bucket edges, so changing how results are
binned never requires re-running inference.

Usage:
    python rebin.py --table <output_dir>/raw_table
    python rebin.py --table <dir>/raw_table --position-bin-size 512
    python rebin.py --table <dir>/raw_table --context-bin-edges 0,8192,32768,131072
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from acceptance_report import (
    DEFAULT_CONTEXT_BIN_EDGES,
    DEFAULT_POSITION_BIN_SIZE,
    load_spec_records,
    write_report,
)

logger = logging.getLogger("evaluate")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        required=True,
        help="Path to a raw_table directory produced by the long-context runner",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the re-bucketed CSVs (default: the table's parent)",
    )
    parser.add_argument(
        "--context-bin-edges",
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
        help=f"By-position bin width in tokens (default: {DEFAULT_POSITION_BIN_SIZE})",
    )
    args = parser.parse_args()

    records = load_spec_records(args.table)
    if not records:
        logger.error("No spec-decode records found in %s", args.table)
        sys.exit(1)
    logger.info("Loaded %d spec-decode records from %s", len(records), args.table)

    context_bin_edges = DEFAULT_CONTEXT_BIN_EDGES
    if args.context_bin_edges:
        context_bin_edges = tuple(
            int(e) for e in args.context_bin_edges.split(",") if e.strip()
        )

    write_report(
        args.output_dir or args.table.parent,
        records,
        context_bin_edges=context_bin_edges,
        position_bin_size=args.position_bin_size,
    )


if __name__ == "__main__":
    main()
