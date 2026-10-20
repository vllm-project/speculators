"""Core evaluation workflows: dataset selection, acceptance, timed sweeps and plots."""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))

import acceptance  # type: ignore[import-not-found]
import request_runner  # type: ignore[import-not-found]
import speedup  # type: ignore[import-not-found]
import throughput_interactivity as ti  # type: ignore[import-not-found]
from acceptance import acceptance_summary  # type: ignore[import-not-found]
from eval_datasets import basic, mrcr  # type: ignore[import-not-found]
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
    parser.add_argument("--max-tokens", type=int, default=16)
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


def _run_args(tmp_path, max_requests=None):
    return acceptance.argparse.Namespace(
        target="http://unused",
        concurrency=2,
        output_dir=tmp_path,
        timeout=1,
        max_requests=max_requests,
    )


def _datasets():
    return [
        {"id": "first:0", "dataset": "first"},
        {"id": "first:1", "dataset": "first"},
        {"id": "second:0", "dataset": "second"},
    ]


def _dataset_results(requests, count):
    return [
        {
            "sample_id": requests[i % len(requests)]["id"],
            "dataset": requests[0]["dataset"],
            "status": "ok",
            "prompt_tokens": 10,
            "metrics": _spec([0, 1], 1)
            if requests[0]["dataset"] == "first"
            else _spec([9, 0], 27),
        }
        for i in range(count)
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
    for flags in [
        ("--data-files", "a.jsonl"),
        ("--split", "test"),
        ("--revision", "pinned"),
        ("--dataset-config", "main"),
    ]:
        with pytest.raises(ValueError, match="single dataset"):
            load_evaluation_requests(_args("speedbench", "custom/data", *flags))
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
    requests = load_evaluation_requests(_args(source, "--messages-column", "prompt"))
    body = requests[0]["body"]
    assert body["messages"] == _conversation()
    assert "answer" not in body
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["stream_options"] == {"include_usage": True}

    # MRCR filters by the served tokenizer before applying the per-dataset cap.
    calls = []

    def rows(*args, **kwargs):
        assert kwargs["data_files"] in (["8needle/*.parquet"], ["2needle/*.parquet"])
        return iter(
            [
                {
                    "prompt": json.dumps(
                        [{**m, "content": str(n)} for m in _conversation()]
                    )
                }
                for n in (250, 20, 40)
            ]
        )

    monkeypatch.setattr(mrcr, "dataset_rows", rows)
    monkeypatch.setattr(
        mrcr.httpx,
        "get",
        lambda *a, **k: httpx.Response(
            200,
            json={"data": [{"id": "test-model", "max_model_len": 100}]},
            request=httpx.Request("GET", "http://server/v1/models"),
        ),
    )
    client_class = httpx.Client

    def tokenize(request):
        body = json.loads(request.content)
        calls.append(body)
        assert [m["role"] for m in body["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert body["add_generation_prompt"]
        assert body["chat_template_kwargs"]["enable_thinking"]
        return httpx.Response(200, json={"count": int(body["messages"][0]["content"])})

    monkeypatch.setattr(
        mrcr.httpx,
        "Client",
        lambda **kw: client_class(transport=httpx.MockTransport(tokenize)),
    )
    args = _args("mrcr", "--max-samples", "1")
    selected = load_evaluation_requests(args)
    assert len(calls) == 2
    assert selected[0]["id"] == "mrcr/8needle:1"
    assert args.dataset_selection["mrcr/8needle"] == {
        "scanned": 2,
        "filtered_by_length": 1,
        "selected": 1,
        "min_prompt_tokens": 0,
        "max_prompt_tokens": 84,
        "server_context": 100,
        "output_budget": 16,
    }
    args = _args(
        "openai/mrcr", "--min-prompt-tokens", "30", "--max-prompt-tokens", "50"
    )
    assert [r["id"] for r in load_evaluation_requests(args)] == ["mrcr/8needle:2"]
    multi = load_evaluation_requests(
        _args("mrcr", "--needles", "2", "8", "--max-samples", "1")
    )
    assert [r["id"] for r in multi] == ["mrcr/2needle:1", "mrcr/8needle:1"]
    with pytest.raises(ValueError, match="fit server context"):
        load_evaluation_requests(_args("mrcr", "--max-prompt-tokens", "100"))


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


def test_fixed_count_does_not_overshoot_with_multiple_workers(tmp_path, monkeypatch):
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
        results = asyncio.run(
            request_runner.collect_requests(
                [{**_request(), "category": "coding"}],
                "http://unused",
                3,
                output,
                10,
                count=5,
            )
        )
    assert sorted(r["sequence"] for r in results) == list(range(5))

    assert all(r["category"] == "coding" for r in results)
    assert all(
        json.loads(line)["category"] == "coding"
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
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

    async def collect(requests, target, concurrency, output, *args, count):
        assert count == len(requests) == 2  # One pass through each dataset.
        if requests[0]["dataset"] == "second":
            assert [r["dataset"] for r in read_csv("summary.csv")] == ["first"]
            assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
            partial = json.loads((tmp_path / "summary.json").read_text())
            assert partial["by_category"]["coding"]["requests"] == 1
            assert read_csv("summary_by_prompt_length.csv")[1]["acceptance_rate"] == ""
        rows = _dataset_results(requests, count)
        for row, request in zip(rows, requests, strict=True):
            row.update(category=request["category"], prompt_tokens=8192)
            if row["category"] == "math":
                row.update(prompt_tokens=8193, metrics=None)
                if row["dataset"] == "second":
                    row["status"] = "error"
            output.write(json.dumps(row) + "\n")
        output.flush()
        return rows

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
    async def collect(requests, target, concurrency, output, *args, count):
        if requests[0]["dataset"] == "second":
            raise RuntimeError("interrupted")
        rows = _dataset_results(requests, count)
        for row in rows:
            output.write(json.dumps(row) + "\n")
        output.flush()
        return rows

    monkeypatch.setattr(acceptance, "collect_requests", collect)
    with pytest.raises(RuntimeError, match="interrupted"):
        acceptance.run(_run_args(tmp_path), _datasets())
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["dataset"] for r in rows] == ["first"]
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["overall"]["requests"] == 2
    limited = tmp_path / "limited"
    limited.mkdir()
    assert acceptance.run(_run_args(limited, max_requests=1), _datasets()) == 0
    summary = json.loads((limited / "summary.json").read_text())
    assert summary["overall"]["requests"] == 1
    assert summary["selected_datasets"] == ["first", "second"]
    assert list(summary["by_dataset"]) == ["first"]


def test_timed_sweep_saves_one_archive_summary_and_plot(tmp_path, monkeypatch):
    monkeypatch.setattr(speedup, "refuse_prefix_caching", lambda target: None)
    monkeypatch.setattr(speedup, "ServerMetrics", _Metrics)
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

    def bench(command, *, check, pass_fds):
        if len(calls) == 2:
            saved = list(csv.DictReader((tmp_path / "summary.csv").open()))
            assert len(saved) == 1
            assert saved[0]["concurrency"] == "1"
            assert saved[0]["repeats"] == "2"
            records = (tmp_path / "benchmarks.jsonl").read_text().splitlines()
            assert len(records) == 6  # Two completed points plus the next start record.
        calls.append(command)
        assert check
        assert len(pass_fds) == 1
        assert command[:2] == ["/existing/guidellm", "run"]
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

    monkeypatch.setattr(speedup.subprocess, "run", bench)
    args = argparse.Namespace(
        target="http://server",
        output_dir=tmp_path,
        model="served",
        label="baseline",
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
        "benchmarks.jsonl",
        "summary.csv",
    ]
    records = [
        json.loads(line)
        for line in (tmp_path / "benchmarks.jsonl").read_text().splitlines()
    ]
    assert records[0]["inputs"] == [{"messages": messages}]
    assert records[0]["copies"] == 5000
    assert records[0]["config"]["max_seconds"] == 100
    records = [records[0], *[r for r in records[1:] if r["status"] == "ok"]]
    assert [row["point"] for row in records[1:]] == [
        "conc1_r1",
        "conc1_r2",
        "conc8_r1",
        "conc8_r2",
    ]
    assert [row["command"] for row in records[1:]] == calls
    assert all(row["report"]["label"] == "baseline" for row in records[1:])
    for command in calls:
        output = Path(json.loads(command[command.index("--output") + 1])["path"])
        data = Path(json.loads(command[command.index("--data") + 1])["path"])
        assert not output.exists()
        assert not data.exists()
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [row["concurrency"] for row in rows] == ["1", "8"]
    for row in rows:
        assert int(row["repeats"]) == 2
        assert float(row["acceptance_length"]) == 3
        assert float(row["waiting_requests_max"]) == 7
        assert row["status"] == "complete"
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


class _Metrics:
    def __init__(self, *args):
        pass

    def start(self):
        pass

    def finish(self):
        return {"waiting_max": 7, "steps": 10, "accepted": 20, "scrape_errors": []}


def test_failed_point_is_archived_without_leaving_temporary_files(
    tmp_path, monkeypatch
):
    # Endpoints are empty at both boundaries, but queueing during the point matters.
    scrapes: list[int] = []

    def metrics(*args, **kwargs):
        i = len(scrapes)
        scrapes.append(i)
        waiting = 7 if i == 1 else 0
        text = f'vllm:num_requests_waiting{{model_name="served"}} {waiting}\n'
        text += 'vllm:num_requests_waiting{model_name="other"} 99\n'
        for key, value in [("num_drafts", 10 + i), ("num_accepted_tokens", 20 + 2 * i)]:
            text += f'vllm:spec_decode_{key}_total{{model_name="served"}} {value}\n'
        return httpx.Response(
            200, text=text, request=httpx.Request("GET", "http://server/metrics")
        )

    monkeypatch.setattr(speedup.httpx, "get", metrics)
    observer = speedup.ServerMetrics("http://server", "served")
    observer.start()
    observer._sample()
    diagnostics = observer.finish()
    assert diagnostics["waiting_max"] == 7
    assert diagnostics["acceptance_length"] == 3
    monkeypatch.setattr(speedup, "refuse_prefix_caching", lambda target: None)
    monkeypatch.setattr(speedup, "ServerMetrics", _Metrics)
    calls: list[list[str]] = []

    def bench(command, *, check, pass_fds):
        calls.append(command)
        path = Path(json.loads(command[command.index("--output") + 1])["path"])
        n = int(
            command[command.index("--profile") + 1].split("streams=")[1].split(",")[0]
        )
        report = timed_report(
            n, errored=[{"error": "HTTP 500"}] if len(calls) == 1 else []
        )
        if len(calls) == 2:
            report["benchmarks"][0]["requests"]["successful"] = []
        path.write_text(json.dumps(report))
        if len(calls) == 1:
            raise speedup.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(speedup.subprocess, "run", bench)
    args = argparse.Namespace(
        output_dir=tmp_path,
        target="http://server",
        streams=[1, 8],
        repeats=2,
        model="served",
        label="test",
        tokenizer=None,
        guidellm_bin="guidellm",
        warmup_seconds=30,
        max_seconds=100,
        resume=False,
        retry_failed=False,
    )
    requests = [
        {"body": {"messages": [{"role": "user", "content": "test"}], "max_tokens": 16}}
    ]
    assert speedup.run(args, requests) == 1
    assert len(calls) == 4
    header, points, _ = speedup.read_archive(tmp_path / "benchmarks.jsonl")
    assert points["conc1_r1"]["status"] == "failed"
    assert points["conc8_r2"]["status"] == "ok"
    assert all(
        not Path(json.loads(c[c.index("--output") + 1])["path"]).exists() for c in calls
    )
    assert [
        p["streams"]
        for p in ti._load_series(argparse.Namespace(series=[str(tmp_path)]))[0][
            "points"
        ]
    ] == [8]
    args.resume = True
    with (tmp_path / "benchmarks.jsonl").open("ab") as handle:
        handle.write(b'{"point":')  # Crash midway through the final journal write.
    assert speedup.run(args, requests) == 1
    assert len(calls) == 4
    args.max_seconds = 101
    with pytest.raises(ValueError, match="configuration or inputs differ"):
        speedup.run(args, requests)
    assert len(calls) == 4
    args.max_seconds = 100
    args.retry_failed = True
    assert speedup.run(args, requests) == 0
    assert len(calls) == 6
    # An interrupted point is retained and skipped until retry is explicitly requested.
    with (tmp_path / "benchmarks.jsonl").open("a") as handle:
        handle.write(json.dumps({"point": "conc8_r2", "status": "started"}) + "\n")
    args.retry_failed = False
    assert speedup.run(args, requests) == 1
    assert len(calls) == 6
    args.retry_failed = True
    assert speedup.run(args, requests) == 0
    assert len(calls) == 7
    assert len(speedup.read_archive(tmp_path / "benchmarks.jsonl")[1]) == 4
    with (tmp_path / "benchmarks.jsonl").open() as locked:
        speedup.fcntl.flock(locked, speedup.fcntl.LOCK_EX | speedup.fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active writer"):
            speedup.run(args, requests)
    assert len(calls) == 7
