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
        --dataset-dir ./speedbench_data
    python evaluate.py --target http://localhost:8000/v1 throughput \\
        --dataset speedbench/qualitative/coding \\
        --dataset-dir ./speedbench_data

    # LongBench-v2 (run prepare_longbench_v2.py once first):
    python evaluate.py --target http://localhost:8000/v1 throughput \\
        --dataset longbench-v2 \\
        --dataset-dir ./longbench_v2_data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    from collections.abc import Callable

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
    parse_request_counts,
    parse_sweep_results,
    print_acceptance_report,
    run_guidellm,
)

logger = logging.getLogger("evaluate")


@dataclass(frozen=True)
class RunSpec:
    """One fully resolved GuideLLM run."""

    label: str
    dataset: str
    data_column_mapper: str
    subset: str | None = None


@dataclass(frozen=True)
class DatasetAdapter:
    """Resolve a prepared benchmark selector into local datasets."""

    resolver: Callable[[str, Path], list[tuple[str, Path]]]
    data_column_mapper: str
    preparation_hint: str


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
# Local benchmark column mappings
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
    prefix, *parts = spec.split("/")
    if prefix != "speedbench" or not parts or not all(parts):
        logger.error("Invalid SPEED-Bench dataset spec: %s", spec)
        sys.exit(1)

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
            "--dataset-dir='%s': no files matching '%s'.\n"
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


def _resolve_longbench_v2(spec: str, data_dir: Path) -> list[tuple[str, Path]]:
    """Resolve data prepared by ``prepare_longbench_v2.py``."""
    if spec != "longbench-v2":
        logger.error("Invalid LongBench-v2 dataset selector: %s", spec)
        sys.exit(1)

    path = data_dir / "longbench_v2.jsonl"
    if not path.is_file():
        logger.error(
            "--dataset-dir='%s': %s not found.\n"
            "Run scripts/evaluate/prepare_longbench_v2.py first.",
            data_dir,
            path.name,
        )
        sys.exit(1)

    return [(spec, path)]


_DATASET_ADAPTERS = {
    "speedbench": DatasetAdapter(
        _resolve_speedbench,
        _SPEEDBENCH_COLUMN_MAPPER,
        "Run scripts/evaluate/prepare_speedbench.py first.",
    ),
    "longbench-v2": DatasetAdapter(
        _resolve_longbench_v2,
        DEFAULT_DATA_COLUMN_MAPPER,
        "Run scripts/evaluate/prepare_longbench_v2.py first.",
    ),
}


def _resolve_runs(args: argparse.Namespace) -> list[RunSpec]:
    """Normalize an HF dataset, local path, or benchmark selector into run specs."""
    spec = args.dataset
    adapter_name = spec.partition("/")[0]
    adapter = _DATASET_ADAPTERS.get(adapter_name)
    if adapter:
        if not args.dataset_dir:
            logger.error(
                "--dataset-dir is required for '%s'.\n%s",
                adapter_name,
                adapter.preparation_hint,
            )
            sys.exit(1)
        return [
            RunSpec(label, str(path), adapter.data_column_mapper)
            for label, path in adapter.resolver(spec, Path(args.dataset_dir))
        ]

    path = Path(spec)
    if path.exists():
        return [RunSpec(path.stem or path.name, spec, args.data_column_mapper)]

    return [
        RunSpec(subset, spec, args.data_column_mapper, subset)
        for subset in (part.strip() for part in args.subsets.split(","))
        if subset
    ]


def _run_subset(
    run: RunSpec,
    args: argparse.Namespace,
    *,
    is_sweep: bool,
    metrics_url: str,
    artifacts_dir: Path,
    output_dir: Path,
    acceptance_csv: CsvWriter | None,
    perf_csv: CsvWriter | None,
) -> tuple[CsvWriter | None, CsvWriter | None, int | None]:
    """Run one normalized benchmark spec."""

    subset = run.label
    logger.info("[%s] Starting", subset)
    safe = subset.replace("/", "_").replace(" ", "_")
    max_tokens = 4096
    guidellm_common = {
        "target": args.target,
        "dataset": run.dataset,
        "data_column_mapper": run.data_column_mapper,
        "max_concurrency": args.max_concurrency,
    }

    if is_sweep:
        gen_len_dir = artifacts_dir / "gen_len"
        gen_len_dir.mkdir(parents=True, exist_ok=True)
        gen_len_output = gen_len_dir / f"gen_len_{safe}.json"
        run_guidellm(
            **guidellm_common,
            subset=run.subset,
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
        key = run.subset or safe
        max_tokens = mapping.get(key, max_tokens)
        logger.info("[%s] max_tokens=%d", subset, max_tokens)

    baseline = _require_metrics(metrics_url)
    profile = "sweep" if is_sweep else "throughput"
    run_output = artifacts_dir / f"run_{safe}.json"
    run_guidellm(
        **guidellm_common,
        subset=run.subset,
        rate=args.sweep_rate if is_sweep else args.gen_len_rate,
        profile=profile,
        max_requests=args.max_requests,
        output_path=run_output,
        max_tokens=max_tokens,
        gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
    )
    current = _require_metrics(metrics_url)

    # Rejected prompts cost the run its sample, not its exit code.
    counts = parse_request_counts(run_output)
    landed = sum(counts.values())
    if landed and counts["requests_ok"] < landed:
        logger.warning(
            "[%s] %d/%d requests did not complete (%d errored, %d incomplete);"
            " metrics below reflect only the %d that did",
            subset,
            landed - counts["requests_ok"],
            landed,
            counts["requests_errored"],
            counts["requests_incomplete"],
            counts["requests_ok"],
        )

    spec = extract_spec_decode_metrics(
        current,
        baseline_metrics=baseline,
    )
    spec.update(counts)
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

    run_items = _resolve_runs(args)

    logger.info(
        "Mode: %s | %d subsets | Output: %s", args.mode, len(run_items), output_dir
    )
    for run in run_items:
        acceptance_csv, perf_csv, mt = _run_subset(
            run,
            args,
            is_sweep=is_sweep,
            metrics_url=metrics_url,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            acceptance_csv=acceptance_csv,
            perf_csv=perf_csv,
        )
        if mt is not None:
            all_max_tokens[run.label] = mt

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
        help=(
            "HF dataset ID, local path, or prepared benchmark selector "
            f"(default: {DEFAULT_DATASET})"
        ),
    )
    parser.add_argument(
        "--subsets",
        default=DEFAULT_SUBSETS,
        help="Comma-separated HF subset names (default: all 9 standard subsets)",
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
        "--dataset-dir",
        default=None,
        help="Prepared data directory for speedbench or longbench-v2",
    )
    parser.add_argument(
        "--speedbench-data-dir",
        dest="dataset_dir",
        help=argparse.SUPPRESS,
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
