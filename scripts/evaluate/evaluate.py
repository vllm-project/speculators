#!/usr/bin/env python3
"""Unified evaluation CLI for speculative decoding benchmarking.

Modes:
    throughput   Max throughput run for acceptance rates
    sweep        Full pipeline (gen-len, sweep, CSV)

Examples:
    python evaluate.py --target http://localhost:8108/v1 throughput
    python evaluate.py --target http://localhost:8000/v1 sweep
    python evaluate.py --target http://localhost:8000/v1 sweep \\
        --subsets "HumanEval,qa" --gen-kwargs '{"temperature":0.6}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from perf_utils import (
    BASE_CSV_COLUMNS,
    CsvWriter,
    acceptance_csv_columns,
    build_backend_args,
    check_dependencies,
    extract_spec_decode_metrics,
    fetch_metrics,
    parse_gen_len_results,
    parse_prometheus_metrics,
    parse_sweep_results,
    print_acceptance_report,
    run_guidellm,
)

logger = logging.getLogger("evaluate")

DEFAULT_DATASET = "RedHatAI/speculator_benchmarks"
DEFAULT_SUBSETS = (
    "HumanEval,math_reasoning,qa,question,rag,"
    "summarization,tool_call,translation,writing"
)
DEFAULT_MAX_CONCURRENCY = 128
DEFAULT_MAX_REQUESTS = 80
DEFAULT_GEN_LEN_RATE = 128
DEFAULT_DATA_COLUMN_MAPPER = '{"text_column":"prompt"}'


def _fetch_model_name(target: str) -> str | None:
    base = target.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    url = f"{base}/models"
    try:
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0].get("id")
    except (URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("Could not fetch model name from %s: %s", url, e)
    return None


def _sanitize_dir_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _require_metrics(metrics_url: str) -> list:
    text = fetch_metrics(metrics_url)
    if text is None:
        logger.error("Failed to fetch metrics from %s", metrics_url)
        sys.exit(1)
    return parse_prometheus_metrics(text)


def _run_subset(
    subset: str,
    args: argparse.Namespace,
    *,
    is_sweep: bool,
    metrics_url: str,
    artifacts_dir: Path,
    output_dir: Path,
    guidellm_common: dict,
    acceptance_csv: CsvWriter | None,
    perf_csv: CsvWriter | None,
) -> tuple[CsvWriter | None, CsvWriter | None, int | None]:
    logger.info("[%s] Starting", subset)
    max_tokens = 4096

    if is_sweep:
        gen_len_dir = artifacts_dir / "gen_len"
        gen_len_dir.mkdir(parents=True, exist_ok=True)
        gen_len_output = gen_len_dir / f"gen_len_{subset}.json"
        run_guidellm(
            **guidellm_common,
            subset=subset,
            profile="throughput",
            max_requests=None,
            output_path=gen_len_output,
            backend_args=build_backend_args(args.gen_kwargs, 4096),
        )
        mapping = parse_gen_len_results(
            [gen_len_output],
            gen_len_dir / f"max_tokens_{subset}.json",
        )
        max_tokens = mapping[subset]
        logger.info("[%s] max_tokens=%d", subset, max_tokens)

    baseline = _require_metrics(metrics_url)
    profile = "sweep" if is_sweep else "throughput"
    run_output = artifacts_dir / f"run_{subset}.json"
    run_guidellm(
        **guidellm_common,
        subset=subset,
        profile=profile,
        max_requests=args.max_requests,
        output_path=run_output,
        backend_args=build_backend_args(args.gen_kwargs, max_tokens),
    )
    current = _require_metrics(metrics_url)

    spec = extract_spec_decode_metrics(
        current,
        baseline_metrics=baseline,
    )
    has_spec = spec and spec.get("num_drafts", 0) > 0

    if has_spec:
        spec["subset"] = subset
        print_acceptance_report(spec)
        if acceptance_csv is None:
            acceptance_csv = CsvWriter(
                output_dir / "acceptance.csv",
                ["subset"] + acceptance_csv_columns(spec),
            )
        acceptance_csv.append(spec)
    else:
        logger.warning("[%s] No speculative decoding metrics found", subset)

    if is_sweep:
        rows = parse_sweep_results(
            run_output,
            spec if has_spec else None,
        )
        if rows:
            if perf_csv is None:
                acc_cols = acceptance_csv_columns(spec) if has_spec else []
                cols = BASE_CSV_COLUMNS + acc_cols
                perf_csv = CsvWriter(
                    output_dir / "perf_results.csv",
                    cols,
                )
            perf_csv.append_rows(rows)

    logger.info("[%s] Complete", subset)
    return acceptance_csv, perf_csv, max_tokens if is_sweep else None


def run_benchmark(args: argparse.Namespace) -> None:
    check_dependencies()
    is_sweep = args.mode == "sweep"

    metrics_url = args.target.rstrip("/").removesuffix("/v1") + "/metrics"
    output_dir = Path(args.output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    subsets = args.subsets.split(",")
    logger.info(
        "Mode: %s | %d subsets | Output: %s",
        args.mode,
        len(subsets),
        output_dir,
    )

    guidellm_common = {
        "target": args.target,
        "dataset": args.dataset,
        "data_column_mapper": args.data_column_mapper,
        "rate": args.gen_len_rate,
        "max_concurrency": args.max_concurrency,
    }

    acceptance_csv = None
    perf_csv = None
    all_max_tokens: dict[str, int] = {}

    for subset in subsets:
        acceptance_csv, perf_csv, mt = _run_subset(
            subset,
            args,
            is_sweep=is_sweep,
            metrics_url=metrics_url,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            guidellm_common=guidellm_common,
            acceptance_csv=acceptance_csv,
            perf_csv=perf_csv,
        )
        if mt is not None:
            all_max_tokens[subset] = mt

    if acceptance_csv is None:
        logger.error("No acceptance metrics collected from any subset")
        sys.exit(1)

    if is_sweep and all_max_tokens:
        with (output_dir / "max_tokens.json").open("w") as f:
            json.dump(all_max_tokens, f, indent=2)

    logger.info("Benchmarking complete! Results: %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Speculative decoding performance evaluation toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python evaluate.py --target http://localhost:8000/v1 throughput\n"
            "  python evaluate.py --target http://localhost:8000/v1 sweep\n"
            "  python evaluate.py --target http://localhost:8000/v1 sweep "
            '--subsets "HumanEval,qa"\n'
        ),
    )
    parser.add_argument(
        "mode",
        choices=["throughput", "sweep"],
        help=(
            "throughput: max-rate run for acceptance rates; "
            "sweep: full benchmarking pipeline"
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="vLLM server endpoint (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset ID or local directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--subsets",
        default=DEFAULT_SUBSETS,
        help="Comma-separated subset names (default: all 9 standard subsets)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model_name>_TIMESTAMP)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Max concurrent requests (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=f"Max requests per sweep point (default: {DEFAULT_MAX_REQUESTS})",
    )
    parser.add_argument(
        "--gen-len-rate",
        type=int,
        default=DEFAULT_GEN_LEN_RATE,
        help=f"Request rate for gen-len estimation (default: {DEFAULT_GEN_LEN_RATE})",
    )
    parser.add_argument(
        "--gen-kwargs",
        default="",
        help="Flat JSON with generation kwargs, e.g. '{\"temperature\":0.6}'",
    )
    parser.add_argument(
        "--data-column-mapper",
        default=DEFAULT_DATA_COLUMN_MAPPER,
        help=f"Column mapping for guidellm (default: {DEFAULT_DATA_COLUMN_MAPPER})",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = _fetch_model_name(args.target)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name:
            logger.info("Detected model: %s", model_name)
            args.output_dir = f"{_sanitize_dir_name(model_name)}_{timestamp}"
        else:
            args.output_dir = f"perf_results_{timestamp}"

    run_benchmark(args)


if __name__ == "__main__":
    main()
