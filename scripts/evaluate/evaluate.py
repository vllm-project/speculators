#!/usr/bin/env python3
"""Unified evaluation CLI for speculative decoding benchmarking.

Modes:
    throughput   Max throughput run for acceptance rates
    sweep        Full pipeline (gen-len, sweep, CSV)

Examples:
    python evaluate.py --target http://localhost:8000/v1 throughput
    python evaluate.py --target http://localhost:8000/v1 sweep
    python evaluate.py --target http://localhost:8000/v1 sweep \\
        --subsets "HumanEval,qa" --gen-kwargs '{"temperature":0.6}'

    # SPEED-Bench (run prepare_speedbench.py once first to split data):
    python evaluate.py --target http://localhost:8000/v1 throughput \\
        --dataset speedbench/qualitative \\
        --speedbench-data-dir ./speedbench_data
    python evaluate.py --target http://localhost:8000/v1 throughput \\
        --dataset speedbench/qualitative/coding \\
        --speedbench-data-dir ./speedbench_data
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
    check_dependencies,
    extract_spec_decode_metrics,
    fetch_metrics,
    parse_gen_kwargs,
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
DEFAULT_MAX_REQUESTS = 200
DEFAULT_GEN_LEN_RATE = 128
DEFAULT_SWEEP_RATE = 10
DEFAULT_DATA_COLUMN_MAPPER = (
    "kind=generative_column_mapper,column_mappings.text_column=prompt"
)

# ---------------------------------------------------------------------------
# SPEED-Bench constants
# ---------------------------------------------------------------------------

_SPEEDBENCH_COLUMN_MAPPER = (
    "kind=generative_column_mapper,column_mappings.text_column=turns"
)


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


def _resolve_speedbench(
    spec: str,
    data_dir: Path,
) -> list[tuple[str, Path]]:
    """Resolve a ``speedbench/<config>[/<category>[/<subcategory>]]`` spec.

    Returns ``(label, path)`` pairs for pre-split JSONL files produced by
    ``scripts/evaluate/prepare_speedbench.py``.  Run that script once after
    NVIDIA's ``prepare.py`` to create per-category/subcategory files.
    """
    parts = spec.removeprefix("speedbench/").split("/")
    config = parts[0]
    rest = parts[1:]
    # Build glob pattern matching prepare_speedbench.py's naming convention:
    #   qualitative/coding          → qualitative_coding*.jsonl
    #   throughput_1k/high_entropy  → throughput_1k_high_entropy_*.jsonl
    #   throughput_1k/high_entropy/code_completion
    #                               → throughput_1k_high_entropy__code_completion*.jsonl
    # Entropy level and subcategory are separated by "__" in the filename.
    if not rest:
        suffix = ""
    elif len(rest) == 1:
        suffix = rest[0]
    else:
        suffix = rest[0] + "__" + "_".join(rest[1:])
    pattern = f"{config}_{suffix}*.jsonl" if suffix else f"{config}_*.jsonl"

    files = sorted(data_dir.glob(pattern))
    if not files:
        logger.error(
            "--speedbench-data-dir='%s': no files matching '%s'.\n"
            "Run scripts/evaluate/prepare_speedbench.py first.",
            data_dir,
            pattern,
        )
        sys.exit(1)

    results = []
    for path in files:
        # Derive label: qualitative_coding → speedbench/qualitative/coding
        stem = path.stem.removeprefix(f"{config}_")
        label = f"speedbench/{config}/{stem.replace('__', '/')}"
        results.append((label, path))
        logger.info("  %s: %s", label, path.name)

    return results


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
    """Run benchmark for one subset.

    *subset* is the human-readable label used for output file names and the
    acceptance CSV.  When ``guidellm_common["dataset"]`` is a local file path
    the ``--data-args`` flag is suppressed automatically; when it is an HF
    dataset ID *subset* is also passed as the data-args filter so guidellm
    loads just that split.
    """

    logger.info("[%s] Starting", subset)
    safe = subset.replace("/", "_").replace(" ", "_")
    max_tokens = 4096

    # For local JSONL files (SPEED-Bench) the dataset path IS the file —
    # no --data-args needed.  For HF datasets subset name doubles as the filter.
    guidellm_subset = None if Path(guidellm_common["dataset"]).exists() else subset

    if is_sweep:
        gen_len_dir = artifacts_dir / "gen_len"
        gen_len_dir.mkdir(parents=True, exist_ok=True)
        gen_len_output = gen_len_dir / f"gen_len_{safe}.json"
        run_guidellm(
            **guidellm_common,
            subset=guidellm_subset,
            profile="throughput",
            rate=args.gen_len_rate,
            max_requests=None,
            output_path=gen_len_output,
            max_tokens=4096,
            gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
        )
        mapping = parse_gen_len_results(
            [gen_len_output],
            gen_len_dir / f"max_tokens_{safe}.json",
        )
        key = guidellm_subset if guidellm_subset else safe
        max_tokens = mapping.get(key, max_tokens)
        logger.info("[%s] max_tokens=%d", subset, max_tokens)

    baseline = _require_metrics(metrics_url)
    profile = "sweep" if is_sweep else "throughput"
    run_output = artifacts_dir / f"run_{safe}.json"
    run_guidellm(
        **guidellm_common,
        subset=guidellm_subset,
        rate=args.sweep_rate if is_sweep else args.gen_len_rate,
        profile=profile,
        max_requests=args.max_requests,
        output_path=run_output,
        max_tokens=max_tokens,
        gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
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

    acceptance_csv = None
    perf_csv = None
    all_max_tokens: dict[str, int] = {}

    # Build a flat list of (label, guidellm_common) to run uniformly.
    # _run_subset auto-detects whether --data-args is needed from the dataset path.
    dataset_spec = args.dataset
    run_items: list[tuple[str, dict]] = []

    if dataset_spec.startswith("speedbench/"):
        if not getattr(args, "speedbench_data_dir", None):
            logger.error(
                "--speedbench-data-dir is required for speedbench/ datasets.\n"
                "Run scripts/evaluate/prepare_speedbench.py first, then add"
                " --speedbench-data-dir <dir>.",
            )
            sys.exit(1)
        pairs = _resolve_speedbench(dataset_spec, Path(args.speedbench_data_dir))
        for label, local_path in pairs:
            run_items.append(
                (
                    label,
                    {
                        "target": args.target,
                        "dataset": str(local_path),
                        "data_column_mapper": _SPEEDBENCH_COLUMN_MAPPER,
                        "max_concurrency": args.max_concurrency,
                    },
                )
            )
    else:
        guidellm_common = {
            "target": args.target,
            "dataset": dataset_spec,
            "data_column_mapper": args.data_column_mapper,
            "max_concurrency": args.max_concurrency,
        }
        for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
            run_items.append((subset, guidellm_common))

    logger.info(
        "Mode: %s | %d subsets | Output: %s", args.mode, len(run_items), output_dir
    )
    for label, common in run_items:
        acceptance_csv, perf_csv, mt = _run_subset(
            label,
            args,
            is_sweep=is_sweep,
            metrics_url=metrics_url,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            guidellm_common=common,
            acceptance_csv=acceptance_csv,
            perf_csv=perf_csv,
        )
        if mt is not None:
            all_max_tokens[label] = mt

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
        "--sweep-rate",
        type=int,
        default=DEFAULT_SWEEP_RATE,
        help=f"Number of sweep rate points (default: {DEFAULT_SWEEP_RATE})",
    )
    parser.add_argument(
        "--gen-kwargs",
        default="",
        help="Flat JSON with generation kwargs, e.g. '{\"temperature\":0.6}'",
    )
    parser.add_argument(
        "--data-column-mapper",
        default=DEFAULT_DATA_COLUMN_MAPPER,
        help="Column mapping for guidellm in typed key=value format"
        f" (default: {DEFAULT_DATA_COLUMN_MAPPER})",
    )
    parser.add_argument(
        "--speedbench-data-dir",
        default=None,
        dest="speedbench_data_dir",
        help=(
            "Path to directory produced by SPEED-Bench prepare.py. "
            "Required when --dataset is a speedbench/ spec."
        ),
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
