"""Core evaluation workflows: dataset selection, acceptance, timed sweeps and plots."""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))

import acceptance  # type: ignore[import-not-found]
import request_runner  # type: ignore[import-not-found]
import speedup  # type: ignore[import-not-found]
import throughput_interactivity as ti  # type: ignore[import-not-found]
from acceptance import acceptance_summary  # type: ignore[import-not-found]
from eval_datasets import basic  # type: ignore[import-not-found]
from loader import (  # type: ignore[import-not-found]
    add_dataset_arguments,
    load_evaluation_requests,
)
from request_runner import send_request  # type: ignore[import-not-found]


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


def timed_report(n=1, *, constraints=None, errored=None):
    def request(end, tpot, tokens=4):
        return {
            "request_end_time": end,
            "inter_token_latency_ms": tpot,
            "output_tokens": tokens,
        }

    return {
        "label": "baseline",
        "benchmarks": [
            {
                "config": {"strategy": {"type_": "concurrent", "streams": n}},
                "scheduler_state": {
                    "end_processing_constraints": constraints or {"max_duration": {}}
                },
                "scheduler_metrics": {
                    "measure_start_time": 30,
                    "measure_end_time": 130,
                },
                "requests": {
                    "successful": [
                        request(20, 999),
                        request(50, 10),
                        request(100, 20),
                        request(110, None, 1),
                        request(131, 999),
                    ],
                    "incomplete": [request(130, None)],
                    "errored": errored or [],
                },
                "metrics": {"output_tokens_per_second": {"total": {"mean": 1234}}},
            }
        ],
    }


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


def test_timed_sweep_saves_reports_summary_and_plot(tmp_path, monkeypatch):
    monkeypatch.setattr(speedup, "refuse_prefix_caching", lambda target: None)
    messages = [
        {"role": "system", "content": "code"},
        {"role": "user", "content": "solve this"},
    ]
    body = {
        "model": "served",
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    calls: list[list[str]] = []

    def bench(command, *, check):
        if len(calls) == 2:
            assert len(list(tmp_path.glob("conc*_r*.json"))) == 2
        calls.append(command)
        assert not check
        assert command[:2] == ["/existing/guidellm", "run"]
        assert not any("max_error" in arg for arg in command)
        assert "--max-requests" not in command
        assert "--num-prompts" not in command
        assert (
            command[command.index("--constraint") + 1]
            == "kind=max_duration,seconds=130"
        )
        profile = command[command.index("--profile") + 1]
        n = int(profile.split("streams=")[1].split(",")[0])
        assert profile.endswith("warmup=30")
        backend = json.loads(command[command.index("--backend") + 1])
        assert backend["max_tokens"] == 8192
        assert backend["extras"]["body"] == {
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.loads(command[command.index("--data") + 1])
        rows = [
            json.loads(line) for line in Path(data["path"]).read_text().splitlines()
        ]
        assert len(rows) == 5000
        assert rows[0]["messages"] == messages
        path = Path(json.loads(command[command.index("--output") + 1])["path"])
        report = timed_report(n)
        # Different repeats must be averaged, including inversion before averaging.
        if len(calls) % 2 == 0:
            benchmark = report["benchmarks"][0]
            benchmark["metrics"]["output_tokens_per_second"]["total"]["mean"] = 2468
            for row in benchmark["requests"]["successful"]:
                if row["inter_token_latency_ms"] is not None:
                    row["inter_token_latency_ms"] *= 2
        path.write_text(json.dumps(report))
        return speedup.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(speedup.subprocess, "run", bench)
    args = argparse.Namespace(
        target="http://server",
        output_dir=tmp_path,
        model="served",
        label="baseline",
        keep_going=False,
        streams=[1, 8],
        repeats=2,
        max_seconds=100,
        warmup_seconds=30,
        guidellm_bin="/existing/guidellm",
        tokenizer="base-tokenizer",
    )
    assert speedup.run(args, [{"body": body}]) == 0
    assert len(calls) == 4
    assert body["messages"] == messages
    assert body["max_tokens"] == 8192
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "conc1_r1.json",
        "conc1_r2.json",
        "conc8_r1.json",
        "conc8_r2.json",
        "summary.csv",
    ]
    for command in calls:
        output = Path(json.loads(command[command.index("--output") + 1])["path"])
        data = Path(json.loads(command[command.index("--data") + 1])["path"])
        assert json.loads(output.read_text())["label"] == "baseline"
        assert not data.exists()
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [row["concurrency"] for row in rows] == ["1", "8"]
    for row in rows:
        assert int(row["repeats"]) == 2
        assert float(row["interactivity_tok_s_per_user"]) == pytest.approx(50)
        assert float(row["output_throughput_tok_s"]) == pytest.approx(1851)
    plotted = ti._load_series(argparse.Namespace(series=[str(tmp_path)]))[0]["points"]
    assert [p["x"] for p in plotted] == pytest.approx([50, 50])
    assert [p["y"] for p in plotted] == pytest.approx([1851, 1851])
    plot = argparse.Namespace(
        series=[str(tmp_path)],
        out=str(tmp_path / "curve.png"),
        title="HumanEval",
        subtitle="",
        note="",
    )
    assert ti.plot(plot) == 0
    assert (tmp_path / "curve.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="already exists"):
        speedup.run(args, [{"body": body}])


def test_failed_points_stop_by_default_and_keep_going_skips_bad_reports(
    tmp_path, monkeypatch, capsys
):
    check_metrics = speedup.refuse_prefix_caching
    monkeypatch.setattr(speedup, "refuse_prefix_caching", lambda target: None)
    calls = []

    def bench(command, *, check):
        assert not check
        assert not any("max_error" in arg for arg in command)
        calls.append(command)
        path = Path(json.loads(command[command.index("--output") + 1])["path"])
        n = int(
            command[command.index("--profile") + 1].split("streams=")[1].split(",")[0]
        )
        if n == 4:
            return speedup.subprocess.CompletedProcess(command, 1)
        report = timed_report(
            n, errored=[{"error": "HTTP 500"}] if n in (1, 64) else []
        )
        if n == 8:
            report["benchmarks"][0]["requests"]["successful"] = []
        path.write_text("invalid JSON" if n == 16 else json.dumps(report))
        return speedup.subprocess.CompletedProcess(command, int(n in (1, 64)))

    monkeypatch.setattr(speedup.subprocess, "run", bench)
    args = argparse.Namespace(
        output_dir=tmp_path,
        target="http://server",
        streams=[1, 4, 8, 16, 32, 64, 128],
        repeats=1,
        keep_going=False,
        model="served",
        label="test",
        tokenizer=None,
        guidellm_bin="guidellm",
        warmup_seconds=30,
        max_seconds=100,
    )
    requests = [
        {"body": {"messages": [{"role": "user", "content": "test"}], "max_tokens": 16}}
    ]
    assert speedup.run(args, requests) == 1
    assert len(calls) == 1
    assert (tmp_path / "conc1_r1.json").exists()
    assert not (tmp_path / "benchmarks.jsonl").exists()
    with pytest.raises(ValueError, match="already exists"):
        speedup.run(args, requests)

    args.output_dir = tmp_path / "keep-going"
    args.output_dir.mkdir()
    args.keep_going = True
    calls.clear()
    assert speedup.run(args, requests) == 1
    assert len(calls) == 7
    assert (args.output_dir / "conc1_r1.json").exists()
    assert (args.output_dir / "conc16_r1.json").read_text() == "invalid JSON"
    assert not (args.output_dir / "conc4_r1.json").exists()
    plotted = ti._load_series(argparse.Namespace(series=[str(args.output_dir)]))[0][
        "points"
    ]
    assert [p["streams"] for p in plotted] == [32, 128]
    rows = list(csv.DictReader((args.output_dir / "summary.csv").open()))
    assert [r["concurrency"] for r in rows] == ["32", "128"]
    assert "Warning: skipping conc16_r1.json" in capsys.readouterr().err

    monkeypatch.setattr(speedup, "refuse_prefix_caching", check_metrics)
    command = calls[-1]
    path = args.output_dir / "conc128_r1.json"
    for metrics in (None, 'vllm:cache_config_info{enable_prefix_caching="true"} 1'):

        def get(*_args, metrics=metrics, **_kwargs):
            if metrics is None:
                raise httpx.ConnectError("metrics unavailable")
            return httpx.Response(
                200, text=metrics, request=httpx.Request("GET", "http://server/metrics")
            )

        monkeypatch.setattr(speedup.httpx, "get", get)
        count = len(calls)
        assert speedup._run_point(command, path, args.target) is False
        assert len(calls) == count
