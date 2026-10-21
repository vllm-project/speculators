"""Core evaluation workflows: dataset selection, acceptance, timed sweeps and plots."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import Mock

import httpx
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))

import acceptance  # type: ignore[import-not-found]
import evaluate  # type: ignore[import-not-found]
import request_runner  # type: ignore[import-not-found]
from acceptance import acceptance_summary  # type: ignore[import-not-found]
from eval_datasets import basic  # type: ignore[import-not-found]
from loader import (  # type: ignore[import-not-found]
    add_dataset_arguments,
    load_evaluation_requests,
)
from request_runner import send_request  # type: ignore[import-not-found]


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
                {"benchmarks": [_benchmark("throughput", successful=0, errored=2)]}
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
def test_invalid_throughput_report_is_fatal(benchmark_args, benchmark_io, write_report):
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
                        _benchmark("throughput", successful=5, sampled_successful=[])
                    ]
                }
            )
        )

    run.side_effect = write_results

    evaluate.run_benchmark(benchmark_args)

    assert (tmp_path / "artifacts" / "run_qa.json").is_file()


def _args(source, *flags):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, nargs="+")
    parser.add_argument("--model", default="test-model")
    parser.add_argument("--target", default="http://server")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=1)
    add_dataset_arguments(parser)
    return parser.parse_args(["--dataset", str(source), *flags])


def _conversation():
    return [
        {"role": "system", "content": "Remember the conversation."},
        {"role": "user", "content": "Find the first answer."},
        {"role": "assistant", "content": "A fixed historical reply."},
        {"role": "user", "content": "Now repeat that answer."},
    ]


def _spec(histogram, drafted):
    return {
        "speculative_decoding": {
            "num_spec_steps": sum(histogram),
            "num_draft_tokens": drafted,
            "num_accepted_draft_tokens": sum(i * n for i, n in enumerate(histogram)),
            "acceptance_histogram": histogram,
        }
    }


def _request():
    return {
        "id": "a",
        "dataset": "test",
        "body": {"model": "test", "stream": True},
    }


def _stream_response(request, complete=True):
    chunks = [
        {
            "choices": [
                {"delta": {"content": "three tokens here"}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"completion_tokens": 3, "prompt_tokens": 4},
            "metrics": _spec([0, 1], 2),
        },
    ]
    text = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if complete:
        text += "data: [DONE]\n\n"
    return httpx.Response(200, text=text, request=request)


def _run_args(tmp_path):
    return acceptance.argparse.Namespace(
        target="http://unused",
        concurrency=2,
        output_dir=tmp_path,
        timeout=1,
    )


def _datasets():
    return [
        {"id": "first:0", "dataset": "first"},
        {"id": "first:1", "dataset": "first"},
        {"id": "second:0", "dataset": "second"},
    ]


def _dataset_results(requests):
    return [
        {
            "sample_id": request["id"],
            "dataset": requests[0]["dataset"],
            "status": "ok",
            "prompt_tokens": 10,
            "metrics": _spec([0, 1], 1)
            if requests[0]["dataset"] == "first"
            else _spec([9, 0], 27),
        }
        for request in requests
    ]


def test_sample_limit_applies_per_dataset_without_overreading(monkeypatch):
    def source(*args, **kwargs):
        for index in range(128):
            yield {"prompt": str(index)}
        raise AssertionError("must stop at the per-dataset limit")

    monkeypatch.setattr(basic, "load_dataset", source)
    args = _args("example/first", "example/second")
    assert args.max_samples == 128
    requests = load_evaluation_requests(args)
    assert len(requests) == 256
    assert [r["dataset"] for r in requests] == ["example/first"] * 128 + [
        "example/second"
    ] * 128
    args.max_samples = 2
    selected = load_evaluation_requests(args)
    assert [r["id"] for r in selected] == [
        "example/first:0",
        "example/first:1",
        "example/second:0",
        "example/second:1",
    ]
    calls = []

    def rows(source, **kwargs):
        calls.append((source, kwargs))
        return iter([{"question": "hello"}])

    monkeypatch.setattr(basic, "dataset_rows", rows)
    load_evaluation_requests(
        _args(
            "gsm8k",
            "--split",
            "train",
            "--revision",
            "rev",
            "--data-files",
            "one.jsonl",
        )
    )
    assert calls[0] == (
        "openai/gsm8k",
        {
            "dataset_config": "main",
            "split": "train",
            "data_files": ["one.jsonl"],
            "revision": "rev",
        },
    )


def test_mrcr_parquet_preserves_the_full_conversation(tmp_path, monkeypatch):
    source = tmp_path / "mrcr.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"prompt": json.dumps(_conversation()), "answer": "not an input"}]
        ),
        source,
    )
    requests = load_evaluation_requests(
        _args(source, "--messages-column", "prompt", "--max-tokens", "16")
    )
    body = requests[0]["body"]
    assert body["messages"] == _conversation()
    assert "answer" not in body
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["stream_options"] == {"include_usage": True}

    def rows(source, **kwargs):
        assert source == "openai/mrcr"
        assert kwargs["data_files"] == ["8needle/*.parquet"]
        return iter([{"prompt": json.dumps(_conversation())}] * 2)

    def unexpected_api_call(*args, **kwargs):
        pytest.fail("MRCR loading must not call model discovery or /tokenize")

    monkeypatch.setattr(basic, "dataset_rows", rows)
    monkeypatch.setattr(httpx, "get", unexpected_api_call)
    monkeypatch.setattr(httpx.Client, "post", unexpected_api_call)
    for source in ("mrcr", "openai/mrcr"):
        args = _args(source, "--data-files", "8needle/*.parquet", "--max-samples", "1")
        selected = load_evaluation_requests(args)
        assert len(selected) == 1
        assert selected[0]["id"] == "mrcr:0"
        assert selected[0]["body"]["messages"] == _conversation()
        assert "max_tokens" not in selected[0]["body"]
        assert "max_completion_tokens" not in selected[0]["body"]
        for flags in (
            ("--max-tokens", "16"),
            ("--body", '{"max_tokens":16}'),
            ("--body", '{"max_completion_tokens":16}'),
        ):
            with pytest.raises(ValueError, match="MRCR does not allow"):
                load_evaluation_requests(_args(source, *flags))

    monkeypatch.setattr(
        basic, "dataset_rows", lambda *a, **kw: iter([{"question": "hello"}])
    )
    assert load_evaluation_requests(_args("gsm8k"))[0]["body"]["max_tokens"] == 8192


def test_speedbench_jsonl_all_null_metadata_then_strings(tmp_path):
    source = tmp_path / "throughput_32k.jsonl"
    rows = [
        {
            "messages": [{"role": "user", "content": "first prompt"}],
            "category": "coding",
            "difficulty": None,
            "unused_metadata": "x" * (11 * 1024 * 1024),
        },
        {
            "messages": [{"role": "user", "content": "second prompt"}],
            "category": "math",
            "difficulty": "hard",
            "unused_metadata": "",
        },
    ]
    # The first row exceeds the Arrow JSON reader's default 10 MiB chunk.
    # Its inferred null-only difficulty column must not reject the next row.
    with source.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    requests = load_evaluation_requests(
        _args("speedbench", "--data-files", str(source))
    )
    assert [r["category"] for r in requests] == ["coding", "math"]
    assert [r["body"]["messages"][0]["content"] for r in requests] == [
        "first prompt",
        "second prompt",
    ]
    assert all("unused_metadata" not in r["body"] for r in requests)


def test_position_rates_count_steps_accepting_through_each_position():
    summary = acceptance_summary(
        [
            {"status": "ok", "metrics": _spec([1, 2, 3, 4], 30)},
            {"status": "ok", "metrics": _spec([10, 0, 0, 0], 10)},
        ]
    )
    rates = summary["acceptance_rate_by_position"]
    assert rates == {"1": 9 / 20, "2": 7 / 20, "3": 4 / 20}
    assert 1 + sum(rates.values()) == pytest.approx(summary["acceptance_length"])


def test_stream_collects_final_metrics_and_rejects_truncation():
    async def run():
        def respond(request):
            if request.url.path.startswith("/bad/"):
                return httpx.Response(
                    400, json={"error": {"message": "maximum context length is 128"}}
                )
            return _stream_response(
                request, complete=request.url.path.startswith("/complete/")
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            complete = await send_request(client, "http://test/complete", _request(), 0)
            truncated = await send_request(
                client, "http://test/truncated", _request(), 1
            )
            bad = await send_request(client, "http://test/bad", _request(), 2)
        return complete, truncated, bad

    complete, truncated, bad = asyncio.run(run())
    assert complete["status"] == "ok"
    assert complete["output_tokens"] == 3
    assert complete["metrics"] == _spec([0, 1], 2)
    assert truncated["status"] == "error"
    assert "incomplete stream" in truncated["error"]
    assert bad["status"] == "error"
    assert "maximum context length is 128" in bad["error"]


def test_selected_requests_are_sent_once_with_multiple_workers(tmp_path, monkeypatch):
    async def fake_send(client, target, request, sequence):
        await asyncio.sleep(0)
        return {
            "sample_id": request["id"],
            "sequence": sequence,
            "status": "ok",
            "request_end_time": 1.0,
        }

    monkeypatch.setattr(request_runner, "send_request", fake_send)
    with (tmp_path / "requests.jsonl").open("w") as output:
        results, consecutive_failures = asyncio.run(
            request_runner.collect_requests(
                [{**_request(), "id": str(i), "category": "coding"} for i in range(5)],
                "http://unused",
                3,
                output,
                10,
            )
        )
    assert sorted(r["sequence"] for r in results) == list(range(5))
    assert sorted(r["sample_id"] for r in results) == [str(i) for i in range(5)]
    assert consecutive_failures == 0

    assert all(r["category"] == "coding" for r in results)
    assert all(
        json.loads(line)["category"] == "coding"
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    )

    statuses = ["error", "error", "ok", "error", "error", "error", "ok"]

    async def failing_send(client, target, request, sequence):
        await asyncio.sleep(0)
        return {"sample_id": request["id"], "status": statuses[sequence]}

    monkeypatch.setattr(request_runner, "send_request", failing_send)
    for concurrency, outcomes in (
        (1, statuses),
        (4, ["error", "error", "error", "ok", "error", "error", "ok"]),
    ):
        statuses = outcomes
        with (tmp_path / f"errors_{concurrency}.jsonl").open("w") as output:
            results, consecutive_failures = asyncio.run(
                request_runner.collect_requests(
                    [{**_request(), "id": str(i)} for i in range(7)],
                    "http://unused",
                    concurrency,
                    output,
                    10,
                )
            )
        assert consecutive_failures == 3
        assert len(results) == 6
        assert [r["sample_id"] for r in results] == [str(i) for i in range(6)]
        assert (
            len((tmp_path / f"errors_{concurrency}.jsonl").read_text().splitlines())
            == 6
        )


def test_acceptance_saves_incremental_dataset_category_and_length_summaries(
    tmp_path, monkeypatch
):
    samples = [
        {"id": f"{dataset}:{i}", "dataset": dataset, "category": category}
        for dataset in ("first", "second")
        for i, category in enumerate(("coding", "math"))
    ]

    def read_csv(name):
        with (tmp_path / name).open() as handle:
            return list(csv.DictReader(handle))

    async def collect(requests, target, concurrency, output, *args):
        assert len(requests) == 2  # One pass through each dataset.
        if requests[0]["dataset"] == "second":
            assert [r["dataset"] for r in read_csv("summary.csv")] == ["first"]
            assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
            partial = json.loads((tmp_path / "summary.json").read_text())
            assert partial["by_category"]["coding"]["requests"] == 1
            assert read_csv("summary_by_prompt_length.csv")[1]["acceptance_rate"] == ""
        rows = _dataset_results(requests)
        for row, request in zip(rows, requests, strict=True):
            row.update(category=request["category"], prompt_tokens=8192)
            if row["category"] == "math":
                row.update(prompt_tokens=8193, metrics=None)
                if row["dataset"] == "second":
                    row["status"] = "error"
            output.write(json.dumps(row) + "\n")
        output.flush()
        return rows, 0

    monkeypatch.setattr(acceptance, "collect_requests", collect)
    assert acceptance.run(_run_args(tmp_path), samples) == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    datasets = read_csv("summary.csv")
    assert [r["dataset"] for r in datasets] == ["first", "second", "overall"]
    assert [int(r["requests"]) for r in datasets] == [2, 2, 4]
    assert all(float(r["elapsed_seconds"]) >= 0 for r in datasets)
    assert report["overall"]["requests"] == 4
    assert report["overall"]["successful_requests"] == 3
    assert report["overall"]["errored_requests"] == 1
    assert report["overall"]["requests_without_spec_metrics"] == 1

    categories = read_csv("summary_by_category.csv")
    lengths = read_csv("summary_by_prompt_length.csv")
    assert [r["category"] for r in categories] == ["coding", "math"]
    assert [r["prompt_length"] for r in lengths] == ["4097-8192", "8193-16384"]
    for row in (datasets[-1], categories[0], lengths[0]):
        # Pool speculative counters, rather than averaging request acceptance rates.
        assert float(row["acceptance_length"]) == pytest.approx(1.1)
        assert float(row["acceptance_rate"]) == pytest.approx(1 / 28)
        assert float(row["acceptance_rate_pos_1"]) == pytest.approx(0.1)
        assert "acceptance_rate_by_position" not in row
    assert int(categories[1]["errored_requests"]) == 1
    assert int(lengths[0]["requests"]) == 2
    assert int(lengths[1]["requests_without_spec_metrics"]) == 1
    assert sum(int(r["requests"]) for r in lengths) == 3
    for rows in (categories, lengths):
        assert rows[1]["acceptance_rate"] == ""
        assert "elapsed_seconds" not in rows[0]
    assert report["by_category"]["coding"]["acceptance_length"] == pytest.approx(1.1)
    assert report["by_prompt_length"]["4097-8192"][
        "acceptance_length"
    ] == pytest.approx(1.1)


def test_interrupted_later_dataset_preserves_completed_csv_row(tmp_path, monkeypatch):
    async def collect(requests, target, concurrency, output, *args):
        if requests[0]["dataset"] == "second":
            raise RuntimeError("interrupted")
        rows = _dataset_results(requests)
        for row in rows:
            output.write(json.dumps(row) + "\n")
        output.flush()
        return rows, 0

    monkeypatch.setattr(acceptance, "collect_requests", collect)
    with pytest.raises(RuntimeError, match="interrupted"):
        acceptance.run(_run_args(tmp_path), _datasets())
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["dataset"] for r in rows] == ["first"]
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"]["requests"] == 2

    async def failing_send(client, target, request, sequence):
        return {"sample_id": request["id"], "status": "error", "error": "HTTP 500"}

    monkeypatch.setattr(request_runner, "send_request", failing_send)
    monkeypatch.setattr(acceptance, "collect_requests", request_runner.collect_requests)
    stopped = tmp_path / "stopped"
    stopped.mkdir()
    args = _run_args(stopped)
    args.concurrency = 1
    assert (
        acceptance.run(args, [*_datasets(), {"id": "third:0", "dataset": "third"}]) == 1
    )
    summary = json.loads((stopped / "summary.json").read_text())
    assert summary["overall"]["errored_requests"] == 3
    assert list(summary["by_dataset"]) == ["first", "second"]
    assert len((stopped / "requests.jsonl").read_text().splitlines()) == 3
