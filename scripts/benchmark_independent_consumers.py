#!/usr/bin/env python3

import argparse
from pathlib import Path

from speculators.benchmarks.independent_consumers import (
    load_config,
    run_benchmark,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run serial 1P1C and independent 1P3C hidden-state benchmarks."
    )
    parser.add_argument("config", type=Path, help="Strict benchmark JSON config")
    parser.add_argument(
        "--run-directory", type=Path, help="New directory for role logs"
    )
    parser.add_argument("--report", type=Path, help="Path for the compact JSON report")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate config without launching"
    )
    parser.add_argument(
        "--scenario",
        choices=("1p1c", "1p3c"),
        help="Run only this scenario; by default both run serially",
    )
    args = parser.parse_args()
    if not args.validate_only and (args.run_directory is None or args.report is None):
        parser.error(
            "--run-directory and --report are required unless --validate-only is set"
        )
    return args


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.validate_only:
        return 0
    if args.run_directory is None or args.report is None:
        raise RuntimeError("benchmark output arguments were not validated")
    report = run_benchmark(config, args.run_directory, scenario_kind=args.scenario)
    write_report(report, args.report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
