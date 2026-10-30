#!/usr/bin/env python3
"""Run simple AIPerf presets or summarize saved acceptance records."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shlex
import subprocess
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _provenance import (
    atomic_write,
    git_sha,
    package_versions,
)
from eval_datasets.speedbench import speedbench_args

logger = logging.getLogger("evaluate")

PROMPT_TOKEN_BOUNDS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)

ALIASES = {
    name: f"spec_al_{name}"
    for name in ("gsm8k", "math500", "humaneval", "mbpp", "mtbench")
}
ALIASES["qualitative"] = "speed_bench_qualitative"

COMMON_DEFAULTS: dict[str, str | None] = {
    "--endpoint-type": "chat",
    "--dataset-sampling-strategy": "sequential",
    "--use-server-token-count": None,
    "--export-level": "raw",
    "--extra-inputs": (
        '{"temperature":1,"chat_template_kwargs":{"enable_thinking":true},'
        '"max_completion_tokens":32768}'
    ),
}
MODE_DEFAULTS: dict[str, dict[str, str | None]] = {
    "acceptance": {"--concurrency": "64", "--request-timeout-seconds": "900"},
    "speedup": {
        "--streaming": None,
        "--auto-plot": None,
        "--concurrency": "1,4,8,16,32,64,128",
        "--num-profile-runs": "3",
        "--no-profile-run-disable-warmup-after-first": None,
        "--warmup-duration": "30",
        "--benchmark-duration": "100",
        "--benchmark-grace-period": "0",
    },
}


def save_eval_provenance(output_dir: Path) -> None:
    """Write the invocation, source revision, and package versions."""
    try:
        sha = git_sha(Path(__file__).resolve().parents[2])
        header = "\n".join(
            [
                f"# Timestamp: {datetime.now(UTC).isoformat()}",
                f"# Git SHA: {sha}",
                *package_versions(),
            ]
        )
        atomic_write(
            output_dir / "eval_command.txt",
            f"{header}\n{shlex.join(sys.argv)}\n",
        )
    except OSError:
        logger.warning("Failed to save eval_command.txt", exc_info=True)


def acceptance_row(metrics: list[dict | None]) -> dict:
    """Use pooled histograms for acceptance length and per-position rates."""
    histogram: Counter[int] = Counter()
    drafts = positions = 0
    for spec in metrics:
        if not spec:
            continue
        histogram.update({int(k): n for k, n in spec["acceptance_histogram"].items()})
        drafts += spec["num_draft_tokens"]
        positions = max(positions, spec.get("num_spec_tokens") or 0)
    steps = sum(histogram.values())
    accepted = sum(k * count for k, count in histogram.items())

    row = {
        "acceptance_length": round(1 + accepted / steps, 4) if steps else "",
        "acceptance_rate_pct": round(100 * accepted / drafts, 2) if drafts else "",
    }
    for position in range(1, max(positions, max(histogram, default=0)) + 1):
        accepted_steps = sum(count for k, count in histogram.items() if k >= position)
        row[f"acceptance_rate_pos_{position}_pct"] = (
            round(100 * accepted_steps / steps, 2) if steps else ""
        )
    return row


def prompt_bin(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    index = bisect_left(PROMPT_TOKEN_BOUNDS, tokens)
    if index == len(PROMPT_TOKEN_BOUNDS):
        return f">{PROMPT_TOKEN_BOUNDS[-1]}"
    low = PROMPT_TOKEN_BOUNDS[index - 1] + 1 if index else 0
    return f"{low}-{PROMPT_TOKEN_BOUNDS[index]}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def summarize_acceptance_run(
    report: dict, path: Path, root: Path
) -> tuple[list[dict], list[dict], bool]:
    """Calculate one overall row and one row per prompt-length bucket."""
    metrics = []
    buckets = defaultdict(list)
    valid = True
    with path.with_name("profile_export.jsonl").open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record["metadata"]["benchmark_phase"] != "profiling":
                    continue
                failed = bool(record.get("error"))
                token_metric = "error_isl" if failed else "input_sequence_length"
                tokens = record["metrics"].get(token_metric, {}).get("value")
                bucket = prompt_bin(tokens)
                spec = None if failed else record.get("spec_decode_acceptance")
                metrics.append(spec)
                buckets[bucket].append(spec)
            except (ValueError, KeyError, TypeError) as exc:
                logger.error("%s: %s", path, exc)
                valid = False

    count = report["request_count"]["avg"]
    if (
        sum(bool(spec) for spec in metrics) != count
        or report.get("spec_decode_acceptance_length", {}).get("count", 0) != count
    ):
        logger.error("%s: missing metrics or incomplete request export", path)
        valid = False
    if not metrics:
        return [], [], False

    name = path.parent.relative_to(root).as_posix()
    overall = [{"dataset": name, **acceptance_row(metrics)}]
    by_length = [
        {
            "dataset": name,
            "prompt_length": bucket,
            "requests": len(group),
            **acceptance_row(group),
        }
        for bucket, group in buckets.items()
    ]
    return overall, by_length, valid


def summarize_output_directory(output_dir: Path, mode: str) -> bool:
    """Validate native runs and write the two additional acceptance CSVs."""
    paths = [
        path
        for path in sorted(output_dir.rglob("profile_export_aiperf.json"))
        if "phases" not in path.relative_to(output_dir).parts
    ]
    if not paths:
        logger.error("No AIPerf summaries found in %s", output_dir)
    valid = bool(paths)
    overall_rows = []
    length_rows = []
    for path in paths:
        try:
            report = json.loads(path.read_text())
            count = report["request_count"]["avg"]
            errors = report.get("error_request_count", {}).get("avg", 0)
            complete = report["is_complete"] and not report["was_cancelled"]
            if not complete or count <= 0 or errors:
                logger.error("%s: incomplete run or %s request errors", path, errors)
                valid = False
            if mode == "acceptance":
                overall, by_length, records_valid = summarize_acceptance_run(
                    report, path, output_dir
                )
                overall_rows.extend(overall)
                length_rows.extend(by_length)
                valid = valid and records_valid
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("%s: %s", path, exc)
            valid = False
    write_csv(output_dir / "acceptance.csv", overall_rows)
    write_csv(output_dir / "acceptance_by_prompt_length.csv", length_rows)
    return valid


def dataset_args(name: str, args: argparse.Namespace) -> list[str]:
    """Resolve dataset aliases and dispatch to the appropriate loader."""
    name = ALIASES.get(name, name).replace("-", "_")
    if name.startswith("speed_bench_"):
        return speedbench_args(name, args)
    return ["--public-dataset", name]


def run_profile(args: argparse.Namespace, profile_args: list[str], output: Path) -> int:
    """Run AIPerf with preset defaults and explicit CLI overrides."""
    supplied = {arg.split("=", 1)[0].replace("--no-", "--", 1) for arg in profile_args}
    defaults = COMMON_DEFAULTS | MODE_DEFAULTS[args.mode]
    if args.num_profile_runs == 1:
        defaults.pop("--no-profile-run-disable-warmup-after-first", None)
    command = ["aiperf", "profile", *profile_args]
    for flag, value in defaults.items():
        if flag.replace("--no-", "--", 1) not in supplied:
            command.append(flag)
            if value is not None:
                command.append(value)
    command.extend(["--output-artifact-dir", str(output)])
    (output / "aiperf_command.txt").write_text(shlex.join(command) + "\n")
    return subprocess.run(command, check=False).returncode  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Other flags pass to 'aiperf profile'. Use canonical long names.",
    )
    parser.add_argument("mode", choices=(*MODE_DEFAULTS, "summarize"))
    parser.add_argument("--output-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        nargs="+",
        help=(
            "Datasets to run sequentially (default: qualitative for acceptance, "
            "humaneval for speedup)."
        ),
    )
    args, profile_args = parser.parse_known_args(argv)
    if args.mode == "summarize":
        return int(
            not summarize_output_directory(args.output_artifact_dir, "acceptance")
        )
    native_only = {"--config", "-f", "--conversation-num", "--num-sessions"}
    if any(arg.split("=", 1)[0] in native_only for arg in profile_args):
        parser.error("Use canonical flags here; run 'aiperf profile' for YAML.")
    # Inspect the options we need without changing the forwarded arguments.
    profile_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    profile_parser.add_argument("--input-file", type=Path)
    profile_parser.add_argument("--request-count", type=int)
    profile_parser.add_argument("--num-conversations", type=int)
    profile_parser.add_argument("--benchmark-duration", type=float)
    profile_parser.add_argument("--num-profile-runs", type=int)
    args, _ = profile_parser.parse_known_args(profile_args, namespace=args)
    default_dataset = "qualitative" if args.mode == "acceptance" else "humaneval"
    datasets = list(dict.fromkeys(args.dataset or [default_dataset]))
    args.output_artifact_dir.mkdir(parents=True, exist_ok=False)
    save_eval_provenance(args.output_artifact_dir)
    exit_code = 0
    for dataset in datasets:
        output = args.output_artifact_dir / dataset
        output.mkdir()
        run_args = [*profile_args, *dataset_args(dataset, args)]
        returncode = run_profile(args, run_args, output)
        exit_code = exit_code or returncode
    valid = summarize_output_directory(args.output_artifact_dir, args.mode)
    return exit_code or int(not valid)


if __name__ == "__main__":
    sys.exit(main())
