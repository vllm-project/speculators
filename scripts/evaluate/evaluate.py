#!/usr/bin/env python3
"""Evaluate acceptance or collect speedup curves from datasets.

    python scripts/evaluate/evaluate.py acceptance --help
    python scripts/evaluate/evaluate.py speedup --help

Use throughput_interactivity.py plot to compare speedup result directories offline.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from acceptance import run as run_acceptance
from loader import add_dataset_arguments, load_evaluation_requests
from speedup import run as run_speedup

from speculators.provenance import (
    atomic_write,
    find_package_repo,
    git_sha,
    package_versions,
)

logger = logging.getLogger("evaluate")


def save_eval_provenance(output_dir: Path) -> None:
    """Write ``eval_command.txt`` into *output_dir*.

    Records the full command line, timestamp, and package versions so the
    eval can be reproduced.  Best-effort — never blocks the eval on failure.
    """
    try:
        sha = git_sha(find_package_repo("speculators"))
        versions = package_versions()
        header = "\n".join(
            [
                f"# Timestamp: {datetime.now(UTC).isoformat()}",
                f"# Git SHA: {sha}",
                *versions,
                *package_versions(("guidellm",)),
            ]
        )
        atomic_write(
            output_dir / "eval_command.txt",
            f"{header}\n{shlex.join(sys.argv)}\n",
        )
    except OSError:
        logger.warning("Failed to save eval_command.txt", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("acceptance", "speedup"):
        sub = modes.add_parser(mode)
        sub.add_argument(
            "--dataset",
            required=True,
            nargs="+",
            help="Dataset IDs, local files, or presets: basic, speedbench; "
            "mrcr (acceptance only)",
        )
        add_dataset_arguments(sub)
        sub.add_argument("--target", required=True)
        sub.add_argument("--model", required=True)
        sub.add_argument(
            "--output-dir",
            type=Path,
            help="Output directory (default: MODEL_TIMESTAMP)",
        )
        sub.add_argument(
            "--max-tokens",
            type=int,
            default=8192,
            help="Generation budget per request (default: 8192; overrides --body)",
        )
        if mode == "acceptance":
            sub.add_argument("--timeout", type=float, default=300)
            sub.add_argument(
                "--max-requests",
                type=int,
                help="Request count (default: one pass over the full selection)",
            )
            sub.add_argument(
                "--concurrency",
                type=int,
                default=64,
                help="Concurrent acceptance requests (default: 64)",
            )
        else:
            sub.add_argument(
                "--streams",
                type=lambda value: [int(n) for n in value.split(",")],
                default=[1, 4, 8, 16, 32, 64, 128],
            )
            sub.add_argument("--repeats", type=int, default=3)
            sub.add_argument(
                "--max-seconds",
                type=float,
                default=100,
                help="Measured seconds per point (default: 100)",
            )
            sub.add_argument(
                "--warmup-seconds",
                type=float,
                default=30,
                help="Warmup seconds per point, excluded from metrics (default: 30)",
            )
            sub.add_argument("--guidellm-bin", default="guidellm")
            sub.add_argument(
                "--tokenizer", help="HF tokenizer or local path (default: served model)"
            )
            sub.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    if args.mode == "speedup" and any(
        name in {"mrcr", "openai/mrcr"} for name in args.dataset
    ):
        parser.error("MRCR is currently supported only in acceptance mode")
    if args.mode == "speedup":
        if args.max_seconds <= 0 or args.warmup_seconds < 0:
            parser.error(
                "--max-seconds must be positive and --warmup-seconds nonnegative"
            )
        if 0 < args.warmup_seconds < 1:
            parser.error("--warmup-seconds must be 0 or at least 1")
        if args.repeats <= 0 or any(n <= 0 for n in args.streams):
            parser.error("streams/repeats must be positive")
    args.target = args.target.rstrip("/").removesuffix("/v1")
    if args.output_dir is None:
        model_name = args.model.replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"{model_name}_{timestamp}")
    try:
        requests = load_evaluation_requests(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_eval_provenance(args.output_dir)
        return (
            run_acceptance(args, requests)
            if args.mode == "acceptance"
            else run_speedup(args, requests)
        )
    except (OSError, ValueError, httpx.HTTPError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
