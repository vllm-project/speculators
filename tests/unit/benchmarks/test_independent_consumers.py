from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

import speculators.benchmarks.independent_consumers as benchmark_module
from speculators.benchmarks.independent_consumers import (
    AccountingLedger,
    AccountingProxy,
    BenchmarkConfig,
    ConsumerSpec,
    ConsumerStepEvent,
    ProducerSpec,
    RequestEvent,
    ScenarioSpec,
    analyze_cache_accounting,
    analyze_consumer_steps,
    analyze_producer_common_window,
    analyze_scenario,
    canonical_request_key,
)


def _consumer(consumer_id: str, gpu: int, command: list[str] | None = None):
    return ConsumerSpec(
        consumer_id=consumer_id,
        gpu=gpu,
        command=command or ["python", "trainer.py", "--endpoint", "{endpoint}"],
    )


def _scenario(
    kind: Literal["1p1c", "1p3c"] = "1p3c",
    *,
    multiplicity: int | None = None,
    warmup: int = 1,
    minimum_steady: int = 2,
    minimum_shared: int = 2,
) -> ScenarioSpec:
    consumer_count = 1 if kind == "1p1c" else 3
    return ScenarioSpec(
        kind=kind,
        consumers=[
            _consumer(f"c{index}", index + 1) for index in range(consumer_count)
        ],
        warmup_completions_per_consumer=warmup,
        minimum_steady_completions_per_consumer=minimum_steady,
        minimum_shared_samples=minimum_shared,
        expected_service_completions_per_shared_sample=(
            multiplicity if multiplicity is not None else consumer_count
        ),
    )


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        producer=ProducerSpec(
            gpu=0,
            endpoint="http://127.0.0.1:8000/v1",
            command=["python", "producer.py", "--tensor-parallel-size", "1"],
        ),
        scenarios=[_scenario("1p1c"), _scenario("1p3c")],
        allowed_gpus=[0, 1, 2, 3],
    )


def _event(consumer: str, key: str, completed_at: float, *, valid: bool = True):
    return RequestEvent(
        consumer_id=consumer,
        request_key=key,
        requested_at=completed_at - 0.05,
        completed_at=completed_at,
        status_code=200 if valid else 500,
        valid_request=True,
        valid_completion=valid,
    )


def _valid_events() -> list[RequestEvent]:
    events = []
    times = {
        "c0": [0.10, 0.20, 0.30, 0.40],
        "c1": [0.15, 0.25, 0.35, 0.45],
        "c2": [0.18, 0.28, 0.38, 0.48],
    }
    for consumer, completed_times in times.items():
        for key, completed_at in zip(
            ["warmup", "sample-a", "sample-b", "sample-c"],
            completed_times,
            strict=True,
        ):
            events.append(_event(consumer, key, completed_at))
    return events


def test_config_requires_serial_one_then_three_consumers():
    config = _config()

    assert [scenario.kind for scenario in config.scenarios] == ["1p1c", "1p3c"]
    assert [len(scenario.consumers) for scenario in config.scenarios] == [1, 3]

    with pytest.raises(ValidationError, match="Scenarios must run serially"):
        BenchmarkConfig(
            producer=config.producer,
            scenarios=list(reversed(config.scenarios)),
            allowed_gpus=config.allowed_gpus,
        )


def test_run_benchmark_can_select_only_1p3c(tmp_path, monkeypatch):
    observed = []

    def fake_run_scenario(_config, scenario, _output_dir):
        observed.append(scenario.kind)
        return {"valid": True, "kind": scenario.kind}

    monkeypatch.setattr(benchmark_module, "_run_scenario", fake_run_scenario)

    report = benchmark_module.run_benchmark(
        _config(), tmp_path / "run", scenario_kind="1p3c"
    )

    assert observed == ["1p3c"]
    assert report["selected_scenarios"] == ["1p3c"]
    assert [scenario["kind"] for scenario in report["scenarios"]] == ["1p3c"]


def test_example_config_uses_train_only_workloads():
    config_path = (
        Path(__file__).parents[3]
        / "benchmarks"
        / "independent_consumer_fanout"
        / "config.example.json"
    )
    config = json.loads(config_path.read_text())

    commands = [
        consumer["command"]
        for scenario in config["scenarios"]
        for consumer in scenario["consumers"]
    ]
    assert len(commands) == 4
    for command in commands:
        ratio_index = command.index("--train-data-ratio")
        assert command[ratio_index + 1] == "1.0"


@pytest.mark.parametrize(
    "command",
    [
        ["torchrun", "--nproc-per-node", "3", "trainer.py"],
        ["python", "-m", "torch.distributed.run", "trainer.py"],
        ["python", "trainer.py", "--tensor-parallel-size=3"],
        ["python", "trainer.py", "-tp", "3"],
        ["python", "trainer.py", "--data-parallel-size", "3"],
        ["python", "trainer.py", "--fsdp-shard"],
        ["bash", "trainer.sh"],
    ],
)
def test_config_rejects_distributed_consumer_commands(command):
    with pytest.raises(
        ValidationError,
        match="not an independent|creates one|direct Python|distributed trainer",
    ):
        _consumer("c0", 1, command)


def test_config_rejects_gpu_sharing_and_owned_environment():
    with pytest.raises(ValidationError, match="more than one consumer"):
        ScenarioSpec(
            kind="1p3c",
            consumers=[_consumer("c0", 1), _consumer("c1", 1), _consumer("c2", 2)],
            expected_service_completions_per_shared_sample=3,
        )

    with pytest.raises(ValidationError, match="harness owns"):
        ConsumerSpec(
            consumer_id="c0",
            gpu=1,
            command=["python", "trainer.py"],
            env={"WORLD_SIZE": "3"},
        )


def test_canonical_request_key_is_stable_and_semantic():
    first = json.dumps(
        {"prompt": [1, 2, 3], "max_tokens": 1, "model": "model", "temperature": 0}
    ).encode()
    reordered = json.dumps(
        {"model": "model", "temperature": 1, "max_tokens": 1, "prompt": [1, 2, 3]}
    ).encode()
    changed = json.dumps(
        {"model": "model", "max_tokens": 1, "prompt": [1, 2, 4]}
    ).encode()

    assert canonical_request_key("/v1/completions", first) == canonical_request_key(
        "/v1/completions", reordered
    )
    assert canonical_request_key("/v1/completions", first) != canonical_request_key(
        "/v1/completions", changed
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "model", "max_tokens": 2, "prompt": [1]},
        {"model": "model", "max_tokens": 1, "prompt": [[1]]},
        {"model": "model", "max_tokens": 1, "prompt": [1], "stream": True},
    ],
)
def test_canonical_request_key_rejects_non_fixture_calls(payload):
    with pytest.raises(ValueError):
        canonical_request_key("/v1/completions", json.dumps(payload).encode())


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        self.rfile.read(content_length)
        response = json.dumps(
            {
                "choices": [{"text": ""}],
                "kv_transfer_params": {"hidden_states_path": "/not/reported"},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def test_accounting_proxy_counts_validated_hidden_state_completion():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    ledger = AccountingLedger()
    target = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    proxy = AccountingProxy(target, "c0", ledger, timeout_seconds=2)
    proxy.start()
    try:
        body = json.dumps(
            {"model": "model", "prompt": [1, 2, 3], "max_tokens": 1}
        ).encode()
        request = urllib.request.Request(  # noqa: S310
            proxy.endpoint + "/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
            assert response.status == 200

        [event] = ledger.snapshot()
        assert event.consumer_id == "c0"
        assert event.valid_request
        assert event.valid_completion
        assert event.status_code == 200
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)


def test_analysis_separates_warmup_and_requires_exact_multiplicity():
    result = analyze_scenario(_scenario(), _valid_events(), 0.0, 1.0)

    assert result["valid"]
    assert result["request_accounting"]["requests"] == 12
    assert result["request_accounting"]["qualifying_shared_samples"] == 3
    assert (
        result["request_accounting"]["per_consumer_completions_semantics"]
        == "logical_consumer"
    )
    assert result["steady_state"]["duration_seconds"] == pytest.approx(0.22)
    assert result["steady_state"]["completions"] == 7


def test_analysis_fails_closed_on_missing_consumer_evidence():
    events = [event for event in _valid_events() if event.consumer_id != "c2"]

    result = analyze_scenario(_scenario(), events, 0.0, 1.0)

    assert not result["valid"]
    assert any("c2 completed 0" in reason for reason in result["invalid_reasons"])
    assert any(
        "expected multiplicity" in reason for reason in result["invalid_reasons"]
    )


def test_analysis_fails_closed_on_failed_or_duplicate_completion():
    events = _valid_events()
    events.append(_event("c0", "sample-a", 0.5))
    events[-2] = _event("c2", "sample-c", 0.48, valid=False)

    result = analyze_scenario(_scenario(), events, 0.0, 1.0)

    assert not result["valid"]
    assert result["request_accounting"]["invalid_completions"] == 1
    assert any(
        "ambiguous multiplicity" in reason for reason in result["invalid_reasons"]
    )


def test_publish_once_analysis_accepts_one_service_call_per_shared_sample():
    events = [
        _event("c0", "warmup", 0.1),
        _event("c1", "sample-a", 0.2),
        _event("c2", "sample-b", 0.3),
        _event("c0", "sample-c", 0.4),
    ]

    result = analyze_scenario(
        _scenario(multiplicity=1), events, started_at=0.0, finished_at=1.0
    )

    assert result["valid"]
    assert result["request_accounting"]["requests"] == 4
    assert result["request_accounting"]["qualifying_shared_samples"] == 3
    assert (
        result["request_accounting"]["per_consumer_completions_semantics"]
        == "service_request_owner"
    )
    assert result["steady_state"]["mode"] == "publish_once_service"
    assert (
        result["steady_state"]["completions_per_consumer_semantics"]
        == "service_request_owner"
    )
    assert result["steady_state"]["completions"] == 3


def test_windowed_analysis_reports_bounded_regeneration():
    events = [
        _event("c0", "warmup", 0.1),
        _event("c0", "sample-a", 0.2),
        _event("c2", "sample-a", 0.25),
        _event("c1", "sample-b", 0.3),
        _event("c0", "sample-c", 0.4),
    ]

    result = analyze_scenario(
        _scenario(multiplicity=1),
        events,
        started_at=0.0,
        finished_at=1.0,
        windowed=True,
    )

    assert result["valid"]
    accounting = result["request_accounting"]
    assert accounting["multiplicity_semantics"] == "bounded_window_regeneration"
    assert accounting["observed_multiplicity_histogram"] == {"1": 2, "2": 1}
    assert accounting["regenerated_sample_keys"] == 1
    assert accounting["extra_service_completions"] == 1
    assert result["steady_state"]["mode"] == "bounded_window_service"


def test_windowed_analysis_rejects_more_generations_than_consumers():
    events = [
        _event("c0", "warmup", 0.1),
        _event("c0", "sample-a", 0.2),
        _event("c1", "sample-a", 0.25),
        _event("c2", "sample-a", 0.3),
        _event("c0", "sample-a", 0.35),
        _event("c1", "sample-b", 0.4),
        _event("c2", "sample-c", 0.45),
    ]

    result = analyze_scenario(
        _scenario(multiplicity=1),
        events,
        started_at=0.0,
        finished_at=1.0,
        windowed=True,
    )

    assert not result["valid"]
    assert any(
        "exceed the maximum bounded window multiplicity" in reason
        for reason in result["invalid_reasons"]
    )


def test_producer_common_window_separates_first_publications_and_recaptures():
    events = [
        _event("c0", "sample-a", 0.2),
        _event("c1", "sample-b", 0.35),
        _event("c2", "sample-a", 0.4),
        _event("c0", "sample-c", 0.6),
    ]

    result = analyze_producer_common_window(
        events,
        start_monotonic_ns=300_000_000,
        end_monotonic_ns=500_000_000,
    )

    assert result["valid"]
    assert result["duration_seconds"] == pytest.approx(0.2)
    assert result["requests"] == 2
    assert result["first_publications"] == 1
    assert result["recaptures"] == 1
    assert result["requests_per_second"] == pytest.approx(10.0)
    assert result["effective_unique_samples_per_second"] == pytest.approx(5.0)
    assert result["service_request_owners"] == {"c1": 1, "c2": 1}


def test_producer_common_window_rejects_empty_interval():
    result = analyze_producer_common_window(
        [],
        start_monotonic_ns=500_000_000,
        end_monotonic_ns=500_000_000,
    )

    assert not result["valid"]
    assert result["requests"] == 0
    assert result["requests_per_second"] is None


def _cache_stats(**updates: int) -> dict[str, int]:
    stats = {
        "schema_version": 1,
        "logical_requests": 12,
        "hits": 8,
        "misses": 4,
        "coalesced_waiters": 2,
        "retry_generations": 0,
        "publishes": 4,
        "generation_failures": 0,
        "publish_failures": 0,
        "invalid_artifacts_removed": 0,
        "expired_artifacts_removed": 0,
        "stale_temps_removed": 0,
        "lock_timeouts": 0,
    }
    stats.update(updates)
    return stats


def test_cache_accounting_proves_publish_once_fanout():
    result = analyze_cache_accounting(
        _scenario(multiplicity=1), _cache_stats(), service_completions=4
    )

    assert result["valid"]
    assert result["stats"]["logical_requests"] == 12
    assert result["stats"]["misses"] == result["stats"]["publishes"] == 4
    assert result["stats"]["hits"] == 8


def test_windowed_cache_allows_prefetch_ahead_of_consumer_reads():
    stats = _cache_stats(logical_requests=9, hits=5)

    windowed = analyze_cache_accounting(
        _scenario(multiplicity=1),
        stats,
        service_completions=4,
        windowed=True,
    )
    synchronous = analyze_cache_accounting(
        _scenario(multiplicity=1), stats, service_completions=4
    )

    assert windowed["valid"]
    assert not synchronous["valid"]


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"logical_requests": 11}, "logical_requests"),
        ({"hits": 7}, "cache hits"),
        ({"misses": 3}, "cache misses"),
        ({"publishes": 3}, "cache publishes"),
        ({"retry_generations": 1}, "retry_generations"),
        ({"generation_failures": 1}, "generation_failures"),
        ({"coalesced_waiters": 9}, "coalesced_waiters"),
    ],
)
def test_cache_accounting_fails_closed_on_invalid_counters(updates, reason):
    result = analyze_cache_accounting(
        _scenario(multiplicity=1),
        _cache_stats(**updates),
        service_completions=4,
    )

    assert not result["valid"]
    assert any(reason in message for message in result["invalid_reasons"])


def test_publish_once_requires_cache_accounting():
    publish_once = analyze_cache_accounting(
        _scenario(multiplicity=1), None, service_completions=4
    )
    unshared = analyze_cache_accounting(
        _scenario(multiplicity=3), None, service_completions=12
    )

    assert not publish_once["valid"]
    assert publish_once["invalid_reasons"]
    assert unshared["valid"]


def test_consumer_step_analysis_excludes_warmup_and_reports_percentiles(tmp_path):
    log_path = tmp_path / "consumer.log"
    log_path.write_text(
        "\n".join(
            [
                "profile/step_ms=9.9e+03, global_step=1",
                "profile/step_ms=10.0, global_step=2",
                "profile/step_ms=20.0, global_step=3",
                "profile/step_ms=30.0, global_step=4",
                "profile/step_ms=40.0, global_step=5",
            ]
        )
    )

    result = analyze_consumer_steps(log_path, warmup_steps=1, minimum_steady_steps=4)

    assert result["valid"]
    assert result["observed_steps"] == 5
    assert result["steady_steps"] == 4
    assert result["step_ms_mean"] == 25.0
    assert result["step_ms_p50"] == 20.0
    assert result["step_ms_p95"] == 40.0


def test_consumer_step_analysis_reports_exact_monotonic_measurement_window(tmp_path):
    log_path = tmp_path / "consumer.log"
    log_path.write_text("captured by callback\n")
    events = [
        ConsumerStepEvent(1_000_000_000, 100.0),
        ConsumerStepEvent(2_000_000_000, 20.0),
        ConsumerStepEvent(3_000_000_000, 30.0),
        ConsumerStepEvent(4_000_000_000, 40.0),
    ]

    result = analyze_consumer_steps(
        log_path,
        warmup_steps=1,
        minimum_steady_steps=2,
        events=events,
        measurement_steps=2,
    )

    assert result["valid"]
    assert result["steady_steps"] == 2
    assert result["steady_started_at_monotonic_ns"] == 1_000_000_000
    assert result["steady_finished_at_monotonic_ns"] == 3_000_000_000
    assert result["steady_duration_seconds"] == 2.0
    assert result["steady_steps_per_second"] == 1.0


def test_consumer_step_analysis_fails_closed_on_missing_log(tmp_path):
    result = analyze_consumer_steps(
        tmp_path / "missing.log", warmup_steps=1, minimum_steady_steps=2
    )

    assert not result["valid"]
    assert result["invalid_reasons"] == ["consumer log is missing"]
