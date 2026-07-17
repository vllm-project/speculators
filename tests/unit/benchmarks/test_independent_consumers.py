from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from speculators.benchmarks.independent_consumers import (
    AccountingLedger,
    AccountingProxy,
    BenchmarkConfig,
    ConsumerSpec,
    ProducerSpec,
    RequestEvent,
    ScenarioSpec,
    analyze_consumer_steps,
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
    kind: str = "1p3c",
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


def test_consumer_step_analysis_fails_closed_on_missing_log(tmp_path):
    result = analyze_consumer_steps(
        tmp_path / "missing.log", warmup_steps=1, minimum_steady_steps=2
    )

    assert not result["valid"]
    assert result["invalid_reasons"] == ["consumer log is missing"]
