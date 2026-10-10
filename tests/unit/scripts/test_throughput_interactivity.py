"""Tests for scripts/evaluate/throughput_interactivity.py.

Covers the measurement rules that a review of real sweeps found broken, and the
guards that keep the procedure fixed:
  - per-request statistics exclude requests that finished during warmup
  - /metrics counters are differenced over the measurement window, not the run
  - prometheus_client's `_created` timestamps are not summed into counters
  - a server with prefix caching on is refused before anything runs
  - a damaged sidecar and a fractional warmup
  - check 4 tells a queue inside vLLM from a delay outside it
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "evaluate"
    / "throughput_interactivity.py"
)


@pytest.fixture(scope="module")
def ti():
    spec = importlib.util.spec_from_file_location(
        "throughput_interactivity", _SCRIPT_PATH, submodule_search_locations=[]
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["throughput_interactivity"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("throughput_interactivity", None)


def _request(start: float, end: float, tokens: int, itl_ms: float, ttft_ms: float):
    first_token = start + ttft_ms / 1000.0
    return {
        "request_start_time": start,
        "request_end_time": end,
        "output_tokens": tokens,
        "prompt_tokens": 100,
        "inter_token_latency_ms": itl_ms,
        "time_to_first_token_ms": ttft_ms,
        "info": {
            "timings": {
                "first_token_iteration": first_token,
                "last_token_iteration": end,
                "token_iterations": tokens,
            }
        },
    }


def _benchmark(successful: list, start: float, end: float, streams: int = 2) -> dict:
    return {
        "requests": {"successful": successful, "incomplete": [], "errored": []},
        "scheduler_metrics": {"measure_start_time": start, "measure_end_time": end},
        "config": {"strategy": {"type_": "concurrent", "streams": streams}},
        "scheduler_state": {"end_processing_constraints": {"max_duration": {}}},
    }


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_summarize_excludes_requests_that_finished_during_warmup(ti):
    # window [100, 200): two requests finished in warmup, two inside the window
    warmup = [
        _request(0.0, 50.0, tokens=500, itl_ms=100.0, ttft_ms=50.0),
        _request(50.0, 99.0, tokens=490, itl_ms=100.0, ttft_ms=50.0),
    ]
    inside = [
        _request(90.0, 150.0, tokens=600, itl_ms=10.0, ttft_ms=20.0),
        _request(150.0, 199.0, tokens=200, itl_ms=20.0, ttft_ms=40.0),
    ]
    summary = ti.summarize_benchmark(_benchmark(warmup + inside, 100.0, 200.0))

    assert summary["point"] == "conc2"
    assert summary["streams"] == 2
    assert summary["successful_requests"] == 4
    assert summary["measured_requests"] == 2
    assert summary["mean_itl_ms"] == pytest.approx(15.0)
    assert summary["interactivity_itl_tps_user"] == pytest.approx(1000.0 / 15.0)
    assert summary["mean_output_tokens"] == pytest.approx(400.0)
    assert summary["median_ttft_ms"] == pytest.approx(30.0)
    assert summary["completed_rps"] == pytest.approx(2 / 100.0)
    # throughput still prorates the tokens of the request that straddles the start
    straddling_share = 600 * (150.0 - 100.0) / (150.0 - 90.02)
    assert summary["aggregate_output_tps"] == pytest.approx(
        (straddling_share + 200) / 100.0, rel=1e-6
    )


def test_summarize_rejects_anything_but_a_closed_loop_benchmark(ti):
    benchmark = _benchmark([], 0.0, 10.0)
    benchmark["config"]["strategy"] = {"type_": "constant", "rate": 2.0}
    with pytest.raises(ValueError, match="closed-loop"):
        ti.summarize_benchmark(benchmark)


def test_read_acceptance_tolerates_damaged_sidecars(ti, tmp_path):
    json_path = tmp_path / "conc1_r1.json"
    sidecar = tmp_path / "conc1_r1.metrics.json"
    assert all(math.isnan(v) for v in ti.read_acceptance(json_path).values())
    sidecar.write_text('{"delta": null, "gauges": null}')
    assert all(math.isnan(v) for v in ti.read_acceptance(json_path).values())
    sidecar.write_text(
        json.dumps(
            {
                "delta": {"acceptance_length": 3.5},
                "gauges": {"num_requests_waiting": {"mean": 2.0, "max": 5.0}},
            }
        )
    )
    row = ti.read_acceptance(json_path)
    assert row["acceptance_length"] == 3.5
    assert row["mean_waiting_requests"] == 2.0
    assert math.isnan(row["mean_running_requests"])


# ---------------------------------------------------------------------------
# server metrics
# ---------------------------------------------------------------------------

_METRICS = """\
# HELP vllm:spec_decode_num_drafts_total d
vllm:spec_decode_num_drafts_total{engine="0"} 100.0
vllm:spec_decode_num_drafts_created{engine="0"} 1.79e+09
vllm:spec_decode_num_draft_tokens_total{engine="0"} 700.0
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 300.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 80.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 50.0
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",position="0"} 1.79e+09
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",position="1"} 1.79e+09
vllm:prefix_cache_queries_total{engine="0"} 0.0
vllm:prefix_cache_hits_total{engine="0"} 0.0
vllm:cache_config_info{block_size="16",enable_prefix_caching="False"} 1.0
vllm:num_requests_running{engine="0"} 7.0
vllm:num_requests_waiting{engine="0"} 3.0
vllm:kv_cache_usage_perc{engine="0"} 0.95
"""
_METRICS_CACHING_ON = _METRICS.replace(
    'enable_prefix_caching="False"', 'enable_prefix_caching="True"'
)


def test_spec_decode_counters_skip_created_and_read_gauges(ti):
    counters = ti.spec_decode_counters(_METRICS)
    assert counters["num_drafts"] == 100.0
    assert counters["accepted_per_position"] == [80.0, 50.0]
    assert counters["prefix_cache_queries"] == 0.0
    assert counters["num_requests_running"] == 7.0
    assert counters["num_requests_waiting"] == 3.0
    assert counters["kv_cache_usage_perc"] == 0.95
    # a server without the gauges reports None, not 0
    assert ti.spec_decode_counters("")["num_requests_waiting"] is None


def test_prefix_caching_is_read_from_the_cache_config(ti):
    assert ti.prefix_caching_enabled(_METRICS) is False
    assert ti.prefix_caching_enabled(_METRICS_CACHING_ON) is True
    # no cache_config_info: moving counters are the fallback signal
    no_config = "\n".join(
        line for line in _METRICS.splitlines() if "cache_config" not in line
    )
    assert ti.prefix_caching_enabled(no_config) is None
    moved = no_config.replace(
        'queries_total{engine="0"} 0.0', 'queries_total{engine="0"} 12.0'
    )
    assert ti.prefix_caching_enabled(moved) is True
    assert ti.prefix_caching_enabled("") is None


def _sample(t: float, drafts: float, waiting: float) -> tuple:
    return (
        t,
        {
            "num_drafts": drafts,
            "num_draft_tokens": 7 * drafts,
            "num_accepted_tokens": 3 * drafts,
            "accepted_per_position": [drafts],
            "prefix_cache_queries": 0.0,
            "num_requests_running": 8.0,
            "num_requests_waiting": waiting,
            "kv_cache_usage_perc": None,
        },
    )


def test_window_metrics_differences_counters_over_the_window(ti):
    samples = [_sample(t, drafts=10.0 * t, waiting=float(t)) for t in range(11)]
    metrics = ti.window_metrics(samples, start=3.2, end=7.9)

    assert metrics["window"]["before_sample"] == 3
    assert metrics["window"]["after_sample"] == 8
    assert metrics["delta"]["num_drafts"] == 50.0
    assert metrics["delta"]["acceptance_length"] == pytest.approx(4.0)
    assert metrics["whole_run"]["delta"]["num_drafts"] == 100.0
    waiting = metrics["gauges"]["num_requests_waiting"]
    assert waiting == {"mean": 5.5, "max": 7.0, "n": 4}  # samples at 4, 5, 6, 7
    assert metrics["gauges"]["kv_cache_usage_perc"] is None
    assert len(metrics["series"]) == 11

    whole = ti.window_metrics(samples, None, None)
    assert whole["delta"]["num_drafts"] == 100.0
    assert ti.window_metrics([], 1.0, 2.0) is None


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def _collect_args(tmp_path: Path, *extra: str) -> list[str]:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "a"}\n{"prompt": "b"}\n')
    return [
        "collect",
        "--target",
        "http://127.0.0.1:1",
        "--model",
        "m",
        "--dataset",
        str(prompts),
        "--subset",
        "prompts",
        "--max-tokens",
        "8",
        "--streams",
        "1",
        "--out-dir",
        str(tmp_path / "out"),
        "--label",
        "x",
        *extra,
    ]


def test_collect_dry_run_builds_a_closed_loop_dataset_command(ti, tmp_path, capsys):
    assert ti.main(_collect_args(tmp_path, "--dry-run", "--streams", "2,4")) == 0
    out = capsys.readouterr().out
    assert "=== conc2_r1" in out
    assert "=== conc4_r1" in out
    assert "kind=concurrent,streams=4,warmup=30" in out
    assert "kind=json_file,path=" in out
    assert "/v1/chat/completions" in out
    assert (tmp_path / "out" / "_data" / "prompts_x2500.jsonl").exists()


def test_collect_rejects_a_fractional_warmup(ti, tmp_path):
    with pytest.raises(SystemExit, match="fraction of the run"):
        ti.main(_collect_args(tmp_path, "--dry-run", "--warmup-seconds", "0.5"))
    assert ti.main(_collect_args(tmp_path, "--dry-run", "--warmup-seconds", "0")) == 0


def test_collect_has_no_random_text_or_open_loop_mode(ti, tmp_path):
    for flag in ("--prompt-tokens", "--rates", "--data", "--synchronous"):
        with pytest.raises(SystemExit) as excinfo:
            ti.main(_collect_args(tmp_path, "--dry-run", flag, "1"))
        assert excinfo.value.code == 2, flag  # argparse: unrecognized argument


def test_collect_refuses_a_server_with_prefix_caching(ti, tmp_path, monkeypatch):
    monkeypatch.setattr(ti, "fetch_text", lambda url, timeout=30.0: _METRICS_CACHING_ON)
    monkeypatch.setattr(ti.shutil, "which", lambda name: "/bin/true")
    with pytest.raises(SystemExit, match="prefix caching enabled"):
        ti.main(_collect_args(tmp_path))
    assert not (tmp_path / "out" / "bench_command.txt").exists()  # refused first


def test_collect_needs_vllm_metrics(ti, tmp_path, monkeypatch):
    monkeypatch.setattr(ti.shutil, "which", lambda name: "/bin/true")
    monkeypatch.setattr(ti, "fetch_text", lambda url, timeout=30.0: None)
    with pytest.raises(SystemExit, match="no /metrics"):
        ti.main(_collect_args(tmp_path))
    monkeypatch.setattr(ti, "fetch_text", lambda url, timeout=30.0: "# not vllm\n")
    with pytest.raises(SystemExit, match="enable_prefix_caching"):
        ti.main(_collect_args(tmp_path))


def test_write_provenance_appends_one_block_per_invocation(ti, tmp_path, monkeypatch):
    monkeypatch.setattr(ti, "fetch_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(ti, "_git_sha", lambda path: "abc123")
    args = ti.argparse.Namespace(target="http://127.0.0.1:1")
    ti.write_provenance(tmp_path, args)
    ti.write_provenance(tmp_path, args)
    text = (tmp_path / "bench_command.txt").read_text()
    assert text.count("timestamp: ") == 2
    assert text.count("git_sha: abc123") == 2
    assert "\n\ntimestamp: " in text  # blank line between the blocks


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _row(point: str, streams: int, ttft_ms: float, waiting: float | None) -> dict:
    return {
        "point": point,
        "streams": streams,
        "aggregate_output_tps": 1000.0,
        "mean_active_concurrency": streams,
        "interactivity_itl_tps_user": 100.0,
        "mean_itl_ms": 10.0,
        "median_ttft_ms": ttft_ms,
        "mean_waiting_requests": waiting,
        "mean_running_requests": 8.0,
        "mean_output_tokens": 100.0,
        "completed_rps": 10.0,
        "stop_reason": "max_duration",
    }


def test_check4_tells_a_server_queue_from_a_client_delay(ti, capsys):
    rows = [
        _row("conc1", 1, ttft_ms=80.0, waiting=0.0),
        _row("conc16", 16, ttft_ms=8000.0, waiting=9.0),
        _row("conc32", 32, ttft_ms=9000.0, waiting=0.0),
        _row("conc64", 64, ttft_ms=9500.0, waiting=None),
    ]
    warnings = ti.validate_rows(rows)
    capsys.readouterr()
    queue = [w for w in warnings if "wait inside it" in w]
    outside = [w for w in warnings if "outside the scheduler" in w]
    assert len(queue) == 1
    assert "conc16 (8.0 s, 9 waiting, 8 running)" in queue[0]
    assert "conc64 (9.5 s)" in queue[0]  # no gauge: still blamed on the server
    assert "conc32" not in queue[0]
    assert len(outside) == 1
    assert "conc32 (9.0 s, 0 waiting, 8 running)" in outside[0]
