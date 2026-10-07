"""Regression tests for evaluation output and benchmark completion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))

import evaluate  # type: ignore[import-not-found]


def _benchmark(
    strategy: str,
    rate: float = 0,
    successful: int = 1,
    errored: int = 0,
    sampled_successful: list | None = None,
) -> dict:
    if sampled_successful is None:
        sampled_successful = (
            [{"output_metrics": {"text_tokens": 200}}] if successful else []
        )
    return {
        "config": {"strategy": {"type_": strategy, "rate": rate}},
        "metrics": {
            "requests_per_second": {"successful": {"median": rate}},
            "request_latency": {"successful": {"median": 0.15}},
            "inter_token_latency_ms": {"successful": {"median": 4.2}},
            "time_to_first_token_ms": {"successful": {"median": 18.0}},
            "output_tokens_per_second": {"successful": {"median": 95.0}},
            "output_tokens": {"successful": {"sum": 200}},
            "request_totals": {
                "successful": successful,
                "errored": errored,
                "incomplete": 0,
                "total": successful + errored,
            },
        },
        "requests": {"successful": sampled_successful},
    }


def _spec_metrics(drafts: int, first: int, second: int) -> str:
    return "\n".join(
        [
            f"vllm:spec_decode_num_drafts_total {drafts}",
            f"vllm:spec_decode_num_draft_tokens_total {2 * drafts}",
            f"vllm:spec_decode_num_accepted_tokens_total {first + second}",
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} '
            f"{first}",
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} '
            f"{second}",
        ]
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


@pytest.fixture
def benchmark_args(tmp_path):
    return argparse.Namespace(
        mode="sweep",
        target="http://localhost:8000/v1",
        output_dir=str(tmp_path),
        dataset="RedHatAI/speculator_benchmarks",
        subsets="qa",
        data_column_mapper=evaluate.DEFAULT_DATA_COLUMN_MAPPER,
        max_concurrency=8,
        max_requests=2,
        gen_len_rate=8,
        sweep_rate=2,
        gen_kwargs="",
    )


@pytest.fixture
def benchmark_io(monkeypatch):
    """Fake only external work; exercise real parsers and CSV writers."""

    def write_results(**kwargs):
        benchmarks = [_benchmark("throughput")]
        if kwargs["profile"] == "sweep":
            benchmarks.extend([_benchmark("constant", 1), _benchmark("constant", 5)])
        kwargs["output_path"].write_text(json.dumps({"benchmarks": benchmarks}))

    run = Mock(side_effect=write_results)
    fetch = Mock(return_value="vllm:num_requests_running 0\n")
    monkeypatch.setattr(evaluate, "check_dependencies", lambda: None)
    monkeypatch.setattr(evaluate, "run_guidellm", run)
    monkeypatch.setattr(evaluate, "fetch_metrics", fetch)
    return run, fetch


@pytest.mark.parametrize("mode", ["throughput", "sweep"])
@pytest.mark.parametrize("zero_counters", [False, True])
def test_baseline_completes_without_acceptance(
    benchmark_args, benchmark_io, tmp_path, mode, zero_counters
):
    benchmark_args.mode = mode
    run, fetch = benchmark_io
    if zero_counters:
        fetch.return_value = _spec_metrics(0, 0, 0)

    evaluate.run_benchmark(benchmark_args)

    assert not (tmp_path / "acceptance.csv").exists()
    assert (tmp_path / "artifacts" / "run_qa.json").is_file()
    assert (tmp_path / "eval_command.txt").is_file()
    if mode == "sweep":
        columns, rows = _read_csv(tmp_path / "perf_results.csv")
        assert columns == evaluate.BASE_CSV_COLUMNS
        assert [row["subset"] for row in rows] == ["qa", "qa"]
        assert [float(row["target_rate"]) for row in rows] == [1, 5]
        assert json.loads((tmp_path / "max_tokens.json").read_text()) == {"qa": 256}
        assert run.call_args.kwargs["max_tokens"] == 256
    else:
        assert not (tmp_path / "perf_results.csv").exists()
        assert not (tmp_path / "max_tokens.json").exists()


@pytest.mark.parametrize("mode", ["throughput", "sweep"])
def test_acceptance_is_reported_once_per_subset(
    benchmark_args, benchmark_io, tmp_path, mode
):
    benchmark_args.mode = mode
    benchmark_args.subsets = "qa,HumanEval"
    _, fetch = benchmark_io
    fetch.side_effect = [
        _spec_metrics(10, 8, 4),
        _spec_metrics(30, 26, 12),
        _spec_metrics(30, 26, 12),
        _spec_metrics(40, 32, 14),
    ]

    evaluate.run_benchmark(benchmark_args)

    _, acceptance = _read_csv(tmp_path / "acceptance.csv")
    assert [row["subset"] for row in acceptance] == ["qa", "HumanEval"]
    assert [float(row["num_drafts"]) for row in acceptance] == [20, 10]
    assert [float(row["num_accepted_tokens"]) for row in acceptance] == [26, 8]
    assert [float(row["acceptance_length"]) for row in acceptance] == [2.3, 1.8]
    assert float(acceptance[0]["acceptance_at_pos_0"]) == 0.9
    assert float(acceptance[0]["acceptance_at_pos_1"]) == 0.4

    if mode == "sweep":
        columns, rows = _read_csv(tmp_path / "perf_results.csv")
        assert columns == evaluate.BASE_CSV_COLUMNS + [
            "num_drafts",
            "num_draft_tokens",
            "num_accepted_tokens",
            "acceptance_length",
            "acceptance_at_pos_0",
            "acceptance_at_pos_1",
        ]
        assert [row["subset"] for row in rows] == ["qa", "qa", "HumanEval", "HumanEval"]
        assert [float(row["target_rate"]) for row in rows] == [1, 5, 1, 5]
        assert all(float(row["latency_median_s"]) == 0.15 for row in rows)
        assert [float(row["num_drafts"]) for row in rows] == [20, 20, 10, 10]
        assert [float(row["acceptance_length"]) for row in rows] == [2.3, 2.3, 1.8, 1.8]
        assert json.loads((tmp_path / "max_tokens.json").read_text()) == {
            "qa": 256,
            "HumanEval": 256,
        }


def test_local_dataset_preserves_subset_label(benchmark_args, benchmark_io, tmp_path):
    label = "speedbench/qualitative/coding"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "qualitative_coding.jsonl").write_text('{"turns": "Hello"}\n')
    benchmark_args.dataset = label
    benchmark_args.speedbench_data_dir = str(data_dir)

    evaluate.run_benchmark(benchmark_args)

    _, rows = _read_csv(tmp_path / "perf_results.csv")
    assert [row["subset"] for row in rows] == [label, label]
    assert json.loads((tmp_path / "max_tokens.json").read_text()) == {label: 256}
    run, _ = benchmark_io
    assert all(call.kwargs["subset"] is None for call in run.call_args_list)


@pytest.mark.parametrize("mode", ["throughput", "sweep"])
@pytest.mark.parametrize("failed_snapshot", [0, 1])
def test_metrics_fetch_failure_is_fatal(
    benchmark_args, benchmark_io, mode, failed_snapshot
):
    benchmark_args.mode = mode
    _, fetch = benchmark_io
    snapshots: list[str | None] = ["vllm:num_requests_running 0\n"] * 2
    snapshots[failed_snapshot] = None
    fetch.side_effect = snapshots

    with pytest.raises(SystemExit) as error:
        evaluate.run_benchmark(benchmark_args)
    assert error.value.code == 1


@pytest.mark.parametrize(
    ("mode", "successful_runs"), [("throughput", 0), ("sweep", 0), ("sweep", 1)]
)
def test_guidellm_failure_is_fatal(benchmark_args, benchmark_io, mode, successful_runs):
    benchmark_args.mode = mode
    run, _ = benchmark_io
    write_results = run.side_effect

    def fail_run(**kwargs):
        if run.call_count > successful_runs:
            raise CalledProcessError(1, ["guidellm", "run"])
        write_results(**kwargs)

    run.side_effect = fail_run

    with pytest.raises(CalledProcessError):
        evaluate.run_benchmark(benchmark_args)


@pytest.mark.parametrize("mode", ["throughput", "sweep"])
def test_empty_subset_selection_is_fatal(benchmark_args, benchmark_io, mode):
    benchmark_args.mode = mode
    benchmark_args.subsets = " , "

    with pytest.raises(SystemExit) as error:
        evaluate.run_benchmark(benchmark_args)
    assert error.value.code == 1
    run, _ = benchmark_io
    run.assert_not_called()


def test_sweep_without_load_points_is_fatal(benchmark_args, benchmark_io):
    run, fetch = benchmark_io
    fetch.side_effect = [_spec_metrics(10, 8, 4), _spec_metrics(30, 26, 12)]

    def write_throughput_only(**kwargs):
        kwargs["output_path"].write_text(
            json.dumps({"benchmarks": [_benchmark("throughput")]})
        )

    run.side_effect = write_throughput_only

    with pytest.raises(SystemExit) as error:
        evaluate.run_benchmark(benchmark_args)
    assert error.value.code == 1


def test_zero_successful_throughput_requests_is_fatal(benchmark_args, benchmark_io):
    benchmark_args.mode = "throughput"
    run, _ = benchmark_io

    def write_zero_successful(**kwargs):
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "benchmarks": [
                        _benchmark("throughput", successful=0, errored=2)
                    ]
                }
            )
        )

    run.side_effect = write_zero_successful

    with pytest.raises(SystemExit) as error:
        evaluate.run_benchmark(benchmark_args)
    assert error.value.code == 1


def _write_missing_report(**_kwargs):
    """Never create run_output."""


def _write_empty_report(**kwargs):
    kwargs["output_path"].write_text("")


def _write_malformed_report(**kwargs):
    kwargs["output_path"].write_text(
        json.dumps({"benchmarks": [{"config": {}, "metrics": {}}]})
    )


@pytest.mark.parametrize(
    "write_report",
    [_write_missing_report, _write_empty_report, _write_malformed_report],
    ids=["missing", "empty", "malformed"],
)
def test_invalid_throughput_report_is_fatal(
    benchmark_args, benchmark_io, write_report
):
    benchmark_args.mode = "throughput"
    run, _ = benchmark_io
    run.side_effect = write_report

    with pytest.raises(SystemExit) as error:
        evaluate.run_benchmark(benchmark_args)
    assert error.value.code == 1


def test_throughput_succeeds_with_empty_sampled_requests(
    benchmark_args, benchmark_io, tmp_path
):
    benchmark_args.mode = "throughput"
    run, _ = benchmark_io

    def write_results(**kwargs):
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "benchmarks": [
                        _benchmark(
                            "throughput", successful=5, sampled_successful=[]
                        )
                    ]
                }
            )
        )

    run.side_effect = write_results

    evaluate.run_benchmark(benchmark_args)

    assert (tmp_path / "artifacts" / "run_qa.json").is_file()
