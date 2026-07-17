from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceError(RuntimeError):
    """Raised when a benchmark cannot produce trustworthy evidence."""


_DISTRIBUTED_ENV = {
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
}
_GPU_ENV = "CUDA_VISIBLE_DEVICES"
_DISTRIBUTED_LAUNCHERS = {
    "accelerate",
    "deepspeed",
    "mpiexec",
    "mpirun",
    "srun",
    "torchrun",
}
_PARALLEL_SIZE_OPTIONS = {
    "-dp",
    "-pp",
    "-tp",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data_parallel_size",
    "--nnodes",
    "--nproc-per-node",
    "--nproc_per_node",
    "--pipeline-parallel-size",
    "--pipeline_parallel_size",
    "--sequence-parallel-size",
    "--sequence_parallel_size",
    "--tensor-parallel-size",
    "--tensor_parallel_size",
}


def _option_value(command: list[str], index: int) -> tuple[str, int]:
    token = command[index]
    if "=" in token:
        return token.split("=", 1)[1], index
    if index + 1 >= len(command):
        raise ValueError(f"Missing value for {token}")
    return command[index + 1], index + 1


def _validate_single_process_command(command: list[str]) -> list[str]:
    executable = Path(command[0]).name
    if executable in _DISTRIBUTED_LAUNCHERS:
        raise ValueError(
            f"Distributed launcher {executable!r} is not an independent consumer"
        )
    if command[1:3] == ["-m", "torch.distributed.run"]:
        raise ValueError("torch.distributed.run is not an independent consumer")
    if not executable.startswith("python"):
        raise ValueError("Benchmark roles must be direct Python commands")
    if "--fsdp-shard" in command:
        raise ValueError("--fsdp-shard requires one distributed trainer")

    index = 0
    while index < len(command):
        option = command[index].split("=", 1)[0]
        if option in _PARALLEL_SIZE_OPTIONS:
            raw_value, value_index = _option_value(command, index)
            try:
                value = int(raw_value)
            except ValueError as error:
                raise ValueError(f"{option} must have an integer value") from error
            if value != 1:
                raise ValueError(
                    f"{option}={value} creates one distributed/parallel job, not an "
                    "independent single-GPU consumer"
                )
            index = value_index
        index += 1
    return command


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CommandSpec(_StrictModel):
    command: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if any(not token for token in value):
            raise ValueError("Command arguments must not be empty")
        return _validate_single_process_command(value)

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        owned = (_DISTRIBUTED_ENV | {_GPU_ENV}) & value.keys()
        if owned:
            names = ", ".join(sorted(owned))
            raise ValueError(f"The harness owns these environment variables: {names}")
        return value


class ProducerSpec(CommandSpec):
    gpu: int = Field(ge=0)
    endpoint: str
    startup_timeout_seconds: float = Field(default=900.0, gt=0)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        return value.rstrip("/")


class ConsumerSpec(CommandSpec):
    consumer_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    gpu: int = Field(ge=0)


class ScenarioSpec(_StrictModel):
    kind: Literal["1p1c", "1p3c"]
    consumers: list[ConsumerSpec]
    timeout_seconds: float = Field(default=3600.0, gt=0)
    warmup_completions_per_consumer: int = Field(default=2, ge=0)
    minimum_steady_completions_per_consumer: int = Field(default=5, ge=1)
    warmup_steps_per_consumer: int = Field(default=10, ge=0)
    minimum_steady_steps_per_consumer: int = Field(default=10, ge=1)
    minimum_shared_samples: int = Field(default=5, ge=1)
    expected_service_completions_per_shared_sample: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_consumers(self) -> ScenarioSpec:
        expected_consumers = 1 if self.kind == "1p1c" else 3
        if len(self.consumers) != expected_consumers:
            raise ValueError(
                f"{self.kind} requires exactly {expected_consumers} independent "
                "consumer process(es)"
            )
        ids = [consumer.consumer_id for consumer in self.consumers]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.kind} consumer IDs must be unique")
        gpus = [consumer.gpu for consumer in self.consumers]
        if len(gpus) != len(set(gpus)):
            raise ValueError(f"{self.kind} assigns more than one consumer to a GPU")
        allowed_multiplicities = {1, expected_consumers}
        if self.expected_service_completions_per_shared_sample not in (
            allowed_multiplicities
        ):
            raise ValueError(
                "Expected service multiplicity must be either publish-once (1) or "
                f"one completion per consumer ({expected_consumers})"
            )
        return self


class BenchmarkConfig(_StrictModel):
    producer: ProducerSpec
    scenarios: list[ScenarioSpec] = Field(min_length=2, max_length=2)
    allowed_gpus: list[int] = Field(min_length=2)
    proxy_timeout_seconds: float = Field(default=180.0, gt=0)
    memory_sample_interval_seconds: float = Field(default=0.25, gt=0)

    @field_validator("allowed_gpus")
    @classmethod
    def validate_allowed_gpus(cls, value: list[int]) -> list[int]:
        if any(gpu < 0 for gpu in value):
            raise ValueError("GPU indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("allowed_gpus must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_layout(self) -> BenchmarkConfig:
        if [scenario.kind for scenario in self.scenarios] != ["1p1c", "1p3c"]:
            raise ValueError("Scenarios must run serially in 1p1c, then 1p3c order")
        allowed = set(self.allowed_gpus)
        if self.producer.gpu not in allowed:
            raise ValueError("Producer GPU is outside allowed_gpus")
        for scenario in self.scenarios:
            consumer_gpus = {consumer.gpu for consumer in scenario.consumers}
            if self.producer.gpu in consumer_gpus:
                raise ValueError(
                    f"{scenario.kind} places producer and consumer on the same GPU"
                )
            outside = consumer_gpus - allowed
            if outside:
                raise ValueError(
                    f"{scenario.kind} uses GPUs outside allowed_gpus: {sorted(outside)}"
                )
        return self


@dataclass(frozen=True)
class RequestEvent:
    consumer_id: str
    request_key: str | None
    requested_at: float
    completed_at: float
    status_code: int | None
    valid_request: bool
    valid_completion: bool
    error: str | None = None


class AccountingLedger:
    def __init__(self) -> None:
        self._events: list[RequestEvent] = []
        self._lock = threading.Lock()

    def add(self, event: RequestEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[RequestEvent]:
        with self._lock:
            return list(self._events)


def canonical_request_key(path: str, body: bytes) -> str:
    """Return a stable key for one hidden-state completion request."""
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Request body is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Request JSON must be an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Request is missing a model")
    if payload.get("max_tokens") != 1:
        raise ValueError("Hidden-state requests must use max_tokens=1")
    if payload.get("stream", False):
        raise ValueError("Streaming responses cannot be accounted atomically")

    normalized_path = urlsplit(path).path.rstrip("/")
    if normalized_path.endswith("/chat/completions"):
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Chat request is missing messages")
        identity: dict[str, Any] = {
            "api": "chat.completions",
            "messages": messages,
            "model": model,
        }
    elif normalized_path.endswith("/completions"):
        prompt = payload.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in prompt
            )
        ):
            raise ValueError("Completion prompt must be one non-empty token-ID list")
        identity = {"api": "completions", "model": model, "prompt": prompt}
    else:
        raise ValueError("Request is not a completions endpoint")

    encoded = json.dumps(
        identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_valid_completion(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    transfer = payload.get("kv_transfer_params")
    return bool(
        isinstance(choices, list)
        and choices
        and isinstance(transfer, dict)
        and isinstance(transfer.get("hidden_states_path"), str)
        and transfer["hidden_states_path"]
    )


_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class AccountingProxy:
    """A non-streaming reverse proxy that counts validated service completions."""

    def __init__(
        self,
        target_endpoint: str,
        consumer_id: str,
        ledger: AccountingLedger,
        timeout_seconds: float,
    ) -> None:
        target = urlsplit(target_endpoint)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise ValueError("target_endpoint must be an absolute HTTP(S) URL")
        self._target = target
        self._consumer_id = consumer_id
        self._ledger = ledger
        self._timeout = timeout_seconds
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_type())
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"accounting-proxy-{consumer_id}",
            daemon=True,
        )

    @property
    def endpoint(self) -> str:
        port = self._server.server_address[1]
        base_path = self._target.path.rstrip("/")
        return f"http://127.0.0.1:{port}{base_path}"

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:  # noqa: C901
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_GET(self) -> None:
                self._forward()

            def do_POST(self) -> None:
                self._forward()

            def _forward(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length) if content_length else b""
                is_completion = self.command == "POST" and urlsplit(
                    self.path
                ).path.rstrip("/").endswith("/completions")
                requested_at = time.monotonic()
                request_key = None
                request_error = None
                if is_completion:
                    try:
                        request_key = canonical_request_key(self.path, body)
                    except ValueError as error:
                        request_error = str(error)

                response_status = 502
                response_reason: str | None = None
                response_headers: list[tuple[str, str]] = [
                    ("Content-Type", "application/json")
                ]
                response_body = b'{"error":"accounting proxy upstream failure"}'
                valid_completion = False
                try:
                    connection_type = (
                        http.client.HTTPSConnection
                        if proxy._target.scheme == "https"
                        else http.client.HTTPConnection
                    )
                    port = proxy._target.port or (
                        443 if proxy._target.scheme == "https" else 80
                    )
                    connection = connection_type(
                        proxy._target.hostname, port, timeout=proxy._timeout
                    )
                    headers = {
                        name: value
                        for name, value in self.headers.items()
                        if name.lower() not in _HOP_BY_HOP_HEADERS
                        and name.lower() not in {"host", "accept-encoding"}
                    }
                    headers["Accept-Encoding"] = "identity"
                    connection.request(
                        self.command, self.path, body=body, headers=headers
                    )
                    response = connection.getresponse()
                    response_status = response.status
                    response_reason = response.reason
                    response_body = response.read()
                    response_headers = response.getheaders()
                    connection.close()

                    valid_completion = bool(
                        is_completion
                        and request_key
                        and response.status in range(200, 300)
                        and _is_valid_completion(response_body)
                    )
                except Exception as error:  # noqa: BLE001
                    request_error = request_error or type(error).__name__

                if is_completion:
                    proxy._ledger.add(
                        RequestEvent(
                            consumer_id=proxy._consumer_id,
                            request_key=request_key,
                            requested_at=requested_at,
                            completed_at=time.monotonic(),
                            status_code=response_status,
                            valid_request=request_key is not None,
                            valid_completion=valid_completion,
                            error=request_error,
                        )
                    )

                self.send_response(response_status, response_reason)
                for name, value in response_headers:
                    if name.lower() not in _HOP_BY_HOP_HEADERS:
                        self.send_header(name, value)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(response_body)

        return Handler

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def analyze_scenario(  # noqa: C901
    scenario: ScenarioSpec,
    events: list[RequestEvent],
    started_at: float,
    finished_at: float,
) -> dict[str, Any]:
    """Analyze one scenario and reject incomplete or ambiguous evidence."""
    reasons: list[str] = []
    expected_ids = [consumer.consumer_id for consumer in scenario.consumers]
    by_consumer: dict[str, list[RequestEvent]] = defaultdict(list)
    for event in events:
        by_consumer[event.consumer_id].append(event)

    unexpected_ids = set(by_consumer) - set(expected_ids)
    if unexpected_ids:
        reasons.append(f"events from unknown consumers: {sorted(unexpected_ids)}")
    invalid_requests = sum(not event.valid_request for event in events)
    invalid_completions = sum(not event.valid_completion for event in events)
    if invalid_requests:
        reasons.append(
            f"{invalid_requests} request(s) were not canonical hidden-state calls"
        )
    if invalid_completions:
        reasons.append(f"{invalid_completions} request(s) lacked a valid completion")

    measured: list[RequestEvent] = []
    steady_start = started_at
    steady_end = finished_at
    per_consumer_total: dict[str, int] = {}
    for consumer_id in expected_ids:
        completed = sorted(
            (
                event
                for event in by_consumer.get(consumer_id, [])
                if event.valid_completion
            ),
            key=lambda event: event.completed_at,
        )
        per_consumer_total[consumer_id] = len(completed)
        needed = (
            scenario.warmup_completions_per_consumer
            + scenario.minimum_steady_completions_per_consumer
        )
        if len(completed) < needed:
            reasons.append(
                f"{consumer_id} completed {len(completed)} request(s), "
                f"fewer than {needed}"
            )
            continue
        warmup = scenario.warmup_completions_per_consumer
        if warmup:
            steady_start = max(steady_start, completed[warmup - 1].completed_at)
        steady_end = min(steady_end, completed[-1].completed_at)
        measured.extend(completed[warmup:])

    duration = steady_end - steady_start
    steady_events = [
        event for event in measured if steady_start <= event.completed_at <= steady_end
    ]
    per_consumer_steady = Counter(event.consumer_id for event in steady_events)
    if duration <= 0:
        reasons.append("common steady-state window is empty")
    for consumer_id in expected_ids:
        count = per_consumer_steady[consumer_id]
        if count < scenario.minimum_steady_completions_per_consumer:
            reasons.append(
                f"{consumer_id} has only {count} completion(s) in the common "
                "steady window"
            )

    key_counts = Counter(
        event.request_key for event in measured if event.request_key is not None
    )
    key_consumers: dict[str, set[str]] = defaultdict(set)
    for event in measured:
        if event.request_key is not None:
            key_consumers[event.request_key].add(event.consumer_id)

    multiplicity = scenario.expected_service_completions_per_shared_sample
    if multiplicity == len(expected_ids):
        qualifying = [
            key
            for key, count in key_counts.items()
            if count == multiplicity and key_consumers[key] == set(expected_ids)
        ]
        malformed = [
            key
            for key, count in key_counts.items()
            if count != multiplicity or key_consumers[key] != set(expected_ids)
        ]
    else:
        qualifying = [key for key, count in key_counts.items() if count == 1]
        malformed = [key for key, count in key_counts.items() if count != 1]
    if len(qualifying) < scenario.minimum_shared_samples:
        reasons.append(
            f"only {len(qualifying)} sample key(s) have expected multiplicity "
            f"{multiplicity}; need {scenario.minimum_shared_samples}"
        )
    if malformed:
        reasons.append(
            f"{len(malformed)} measured sample key(s) have ambiguous multiplicity"
        )

    safe_key_counts = {
        key[:16]: count for key, count in sorted(key_counts.items()) if key is not None
    }
    return {
        "valid": not reasons,
        "invalid_reasons": reasons,
        "request_accounting": {
            "requests": len(events),
            "valid_completions": len(events) - invalid_completions,
            "invalid_requests": invalid_requests,
            "invalid_completions": invalid_completions,
            "per_consumer_completions": per_consumer_total,
            "expected_service_completions_per_shared_sample": multiplicity,
            "qualifying_shared_samples": len(qualifying),
            "sample_completion_counts": safe_key_counts,
        },
        "steady_state": {
            "warmup_completions_per_consumer": (
                scenario.warmup_completions_per_consumer
            ),
            "started_at_monotonic": steady_start,
            "finished_at_monotonic": steady_end,
            "duration_seconds": max(duration, 0.0),
            "completions": len(steady_events),
            "completions_per_consumer": dict(per_consumer_steady),
            "completions_per_second": (
                len(steady_events) / duration if duration > 0 else None
            ),
        },
    }


_STEP_TIME_PATTERN = re.compile(
    r"profile/step_ms=(?P<value>[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",
    re.IGNORECASE,
)


def _percentile(values: list[float], percentile: float) -> float:
    index = max(0, min(len(values) - 1, int(len(values) * percentile + 0.999999) - 1))
    return sorted(values)[index]


def analyze_consumer_steps(
    log_path: Path, warmup_steps: int, minimum_steady_steps: int
) -> dict[str, Any]:
    """Extract native trainer step timings and separate warmup from steady state."""
    if not log_path.is_file():
        return {
            "valid": False,
            "invalid_reasons": ["consumer log is missing"],
            "observed_steps": 0,
            "warmup_steps": 0,
            "steady_steps": 0,
            "step_ms_mean": None,
            "step_ms_p50": None,
            "step_ms_p95": None,
        }
    values = []
    with log_path.open(errors="replace") as log_file:
        for line in log_file:
            values.extend(
                float(match.group("value"))
                for match in _STEP_TIME_PATTERN.finditer(line)
            )
    steady = values[warmup_steps:]
    reasons = []
    if len(steady) < minimum_steady_steps:
        reasons.append(
            f"only {len(steady)} steady step timing(s); need {minimum_steady_steps}"
        )
    return {
        "valid": not reasons,
        "invalid_reasons": reasons,
        "observed_steps": len(values),
        "warmup_steps": min(warmup_steps, len(values)),
        "steady_steps": len(steady),
        "step_ms_mean": sum(steady) / len(steady) if steady else None,
        "step_ms_p50": _percentile(steady, 0.50) if steady else None,
        "step_ms_p95": _percentile(steady, 0.95) if steady else None,
    }


@dataclass(frozen=True)
class _GpuSample:
    captured_at: float
    total_memory_mib: dict[int, int]
    role_memory_mib: dict[int, int]
    compute_pids: dict[int, list[int]]


def _run_nvidia_smi(query: str) -> list[str]:
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gpu_snapshot(target_gpus: set[int]) -> _GpuSample:
    gpu_rows = _run_nvidia_smi("gpu=index,uuid,memory.used")
    uuid_to_index: dict[str, int] = {}
    total_memory: dict[int, int] = {}
    for row in gpu_rows:
        raw_index, uuid, raw_memory = (part.strip() for part in row.split(",", 2))
        index = int(raw_index)
        if index in target_gpus:
            uuid_to_index[uuid] = index
            total_memory[index] = int(raw_memory)
    if set(total_memory) != target_gpus:
        missing = sorted(target_gpus - set(total_memory))
        raise EvidenceError(f"nvidia-smi did not report target GPUs {missing}")

    role_memory = dict.fromkeys(target_gpus, 0)
    compute_pids: dict[int, list[int]] = defaultdict(list)
    for row in _run_nvidia_smi("compute-apps=gpu_uuid,pid,used_gpu_memory"):
        uuid, raw_pid, raw_memory = (part.strip() for part in row.split(",", 2))
        index = uuid_to_index.get(uuid)
        if index is None:
            continue
        try:
            pid = int(raw_pid)
            memory = int(raw_memory)
        except ValueError as error:
            raise EvidenceError(f"Unparseable nvidia-smi compute row: {row}") from error
        compute_pids[index].append(pid)
        role_memory[index] += memory
    return _GpuSample(
        captured_at=time.monotonic(),
        total_memory_mib=total_memory,
        role_memory_mib=role_memory,
        compute_pids=dict(compute_pids),
    )


class GpuMonitor:
    def __init__(self, target_gpus: set[int], interval_seconds: float) -> None:
        self._target_gpus = target_gpus
        self._interval = interval_seconds
        self._roots: dict[int, int] = {}
        self._known_pids: dict[int, set[int]] = defaultdict(set)
        self._samples: list[_GpuSample] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._run, name="benchmark-gpu-monitor", daemon=True
        )

    def require_idle(self) -> dict[int, int]:
        sample = _gpu_snapshot(self._target_gpus)
        occupied = {gpu: pids for gpu, pids in sample.compute_pids.items() if pids}
        if occupied:
            raise EvidenceError(
                f"Target GPUs already have compute processes: {occupied}"
            )
        return sample.total_memory_mib

    def set_role_process(self, gpu: int, pid: int) -> None:
        with self._lock:
            self._roots[gpu] = pid
            self._known_pids[gpu].add(pid)

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval * 4))

    def _allowed_pids(self, gpu: int) -> set[int]:
        with self._lock:
            root = self._roots.get(gpu)
            known = set(self._known_pids[gpu])
        if root is None:
            return known
        try:
            descendants = {child.pid for child in psutil.Process(root).children(True)}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = set()
        allowed = known | {root} | descendants
        with self._lock:
            self._known_pids[gpu].update(allowed)
        return allowed

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                sample = _gpu_snapshot(self._target_gpus)
                for gpu, pids in sample.compute_pids.items():
                    if len(pids) > 1:
                        self._errors.append(
                            f"GPU {gpu} has {len(pids)} CUDA compute processes"
                        )
                    foreign = set(pids) - self._allowed_pids(gpu)
                    if foreign:
                        self._errors.append(
                            f"GPU {gpu} has foreign compute PIDs {sorted(foreign)}"
                        )
                self._samples.append(sample)
            except Exception as error:  # noqa: BLE001
                self._errors.append(f"{type(error).__name__}: {error}")

    def summarize(
        self, started_at: float, finished_at: float, baseline: dict[int, int]
    ) -> dict[str, Any]:
        samples = [
            sample
            for sample in self._samples
            if started_at <= sample.captured_at <= finished_at
        ]
        per_gpu: dict[str, Any] = {}
        for gpu in sorted(self._target_gpus):
            role_values = [sample.role_memory_mib[gpu] for sample in samples]
            total_values = [sample.total_memory_mib[gpu] for sample in samples]
            observed = any(sample.compute_pids.get(gpu) for sample in samples)
            per_gpu[str(gpu)] = {
                "baseline_memory_mib": baseline.get(gpu),
                "peak_role_memory_mib": max(role_values) if role_values else None,
                "peak_total_memory_mib": max(total_values) if total_values else None,
                "max_compute_processes": max(
                    (len(sample.compute_pids.get(gpu, [])) for sample in samples),
                    default=0,
                ),
                "compute_process_observed": observed,
            }
        errors = list(dict.fromkeys(self._errors))
        if not samples:
            errors.append("No memory samples fall inside the steady-state window")
        missing = [
            gpu
            for gpu, value in per_gpu.items()
            if not value["compute_process_observed"]
        ]
        if missing:
            errors.append(f"No compute process observed on GPU(s) {missing}")
        return {
            "reliable": not errors,
            "invalid_reasons": errors,
            "sample_count": len(samples),
            "per_gpu": per_gpu,
        }


def _render(values: list[str], replacements: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for original in values:
        value = original
        for name, replacement in replacements.items():
            value = value.replace("{" + name + "}", replacement)
        rendered.append(value)
    return rendered


class _ManagedProcess:
    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        gpu: int,
        log_path: Path,
    ) -> None:
        process_env = os.environ.copy()
        for name in _DISTRIBUTED_ENV:
            process_env.pop(name, None)
        process_env.update(env)
        process_env[_GPU_ENV] = str(gpu)
        self._log = log_path.open("w", encoding="utf-8")
        self.started_at = time.monotonic()
        self.finished_at: float | None = None
        self.process = subprocess.Popen(  # noqa: S603
            command,
            env=process_env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    def close_log(self) -> None:
        self._log.close()

    def terminate(self, grace_seconds: float = 20.0) -> None:
        if self.process.poll() is not None:
            self.finished_at = self.finished_at or time.monotonic()
            self.close_log()
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
        self.finished_at = time.monotonic()
        self.close_log()


def _wait_for_producer(
    endpoint: str, process: subprocess.Popen[str], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    url = endpoint.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise EvidenceError(
                f"Producer exited during startup with code {return_code}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                if response.status in range(200, 300):
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise EvidenceError(f"Producer was not ready after {timeout:.1f} seconds")


def _run_scenario(  # noqa: C901
    config: BenchmarkConfig, scenario: ScenarioSpec, output_dir: Path
) -> dict[str, Any]:
    scenario_dir = output_dir / scenario.kind
    scenario_dir.mkdir(parents=True, exist_ok=False)
    target_gpus = {
        config.producer.gpu,
        *(consumer.gpu for consumer in scenario.consumers),
    }
    monitor = GpuMonitor(target_gpus, config.memory_sample_interval_seconds)
    baseline: dict[int, int] = {}
    ledger = AccountingLedger()
    producer: _ManagedProcess | None = None
    consumers: list[_ManagedProcess] = []
    proxies: list[AccountingProxy] = []
    runtime_errors: list[str] = []
    started_at = time.monotonic()
    finished_at = started_at

    producer_replacements = {
        "output_dir": str(scenario_dir / "producer"),
        "scenario": scenario.kind,
    }
    Path(producer_replacements["output_dir"]).mkdir()
    try:
        baseline = monitor.require_idle()
        producer = _ManagedProcess(
            _render(config.producer.command, producer_replacements),
            {
                key: _render([value], producer_replacements)[0]
                for key, value in config.producer.env.items()
            },
            config.producer.gpu,
            scenario_dir / "producer.log",
        )
        monitor.set_role_process(config.producer.gpu, producer.process.pid)
        _wait_for_producer(
            config.producer.endpoint,
            producer.process,
            config.producer.startup_timeout_seconds,
        )

        for consumer in scenario.consumers:
            proxy = AccountingProxy(
                config.producer.endpoint,
                consumer.consumer_id,
                ledger,
                config.proxy_timeout_seconds,
            )
            proxy.start()
            proxies.append(proxy)

        monitor.start()
        started_at = time.monotonic()
        for consumer, proxy in zip(scenario.consumers, proxies, strict=True):
            consumer_dir = scenario_dir / consumer.consumer_id
            consumer_dir.mkdir()
            replacements = {
                "consumer_id": consumer.consumer_id,
                "endpoint": proxy.endpoint,
                "output_dir": str(consumer_dir),
                "scenario": scenario.kind,
            }
            process = _ManagedProcess(
                _render(consumer.command, replacements),
                {
                    **{
                        key: _render([value], replacements)[0]
                        for key, value in consumer.env.items()
                    },
                    "SPECULATORS_CONSUMER_ID": consumer.consumer_id,
                },
                consumer.gpu,
                scenario_dir / f"{consumer.consumer_id}.log",
            )
            consumers.append(process)
            monitor.set_role_process(consumer.gpu, process.process.pid)

        deadline = started_at + scenario.timeout_seconds
        for consumer, process in zip(scenario.consumers, consumers, strict=True):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EvidenceError(f"{scenario.kind} exceeded its timeout")
            try:
                return_code = process.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                raise EvidenceError(
                    f"{consumer.consumer_id} exceeded the scenario timeout"
                ) from error
            if return_code != 0:
                raise EvidenceError(
                    f"{consumer.consumer_id} exited with code {return_code}"
                )
            process.finished_at = time.monotonic()
        producer_return_code = producer.process.poll()
        if producer_return_code is not None:
            raise EvidenceError(
                f"Producer exited unexpectedly with code {producer_return_code}"
            )
        finished_at = time.monotonic()
    except Exception as error:  # noqa: BLE001
        runtime_errors.append(f"{type(error).__name__}: {error}")
        finished_at = time.monotonic()
    finally:
        monitor.stop()
        for process in consumers:
            process.terminate()
        for proxy in proxies:
            proxy.close()
        if producer is not None:
            producer.terminate()

    analysis = analyze_scenario(scenario, ledger.snapshot(), started_at, finished_at)
    consumer_steps = {}
    for consumer in scenario.consumers:
        step_result = analyze_consumer_steps(
            scenario_dir / f"{consumer.consumer_id}.log",
            scenario.warmup_steps_per_consumer,
            scenario.minimum_steady_steps_per_consumer,
        )
        consumer_steps[consumer.consumer_id] = step_result
        runtime_errors.extend(
            f"{consumer.consumer_id}: {reason}"
            for reason in step_result["invalid_reasons"]
        )
    steady = analysis["steady_state"]
    memory = monitor.summarize(
        steady["started_at_monotonic"],
        steady["finished_at_monotonic"],
        baseline,
    )
    invalid_reasons = [
        *runtime_errors,
        *analysis["invalid_reasons"],
        *memory["invalid_reasons"],
    ]
    return {
        "kind": scenario.kind,
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "consumer_processes": [
            {
                "consumer_id": consumer.consumer_id,
                "gpu": consumer.gpu,
                "return_code": process.process.returncode,
                "runtime_seconds": (
                    process.finished_at - process.started_at
                    if process.finished_at is not None
                    else None
                ),
            }
            for consumer, process in zip(scenario.consumers, consumers, strict=False)
        ],
        "request_accounting": analysis["request_accounting"],
        "steady_state": steady,
        "consumer_step_times": consumer_steps,
        "makespan_seconds": max(finished_at - started_at, 0.0),
        "memory": memory,
    }


def _redacted_config(config: BenchmarkConfig) -> dict[str, Any]:
    value = config.model_dump(mode="json")
    value["producer"]["env"] = sorted(value["producer"]["env"])
    for scenario in value["scenarios"]:
        for consumer in scenario["consumers"]:
            consumer["env"] = sorted(consumer["env"])
    return value


def run_benchmark(config: BenchmarkConfig, output_dir: Path) -> dict[str, Any]:
    """Run fresh-producer 1P1C and 1P3C scenarios and return one JSON report."""
    output_dir.mkdir(parents=True, exist_ok=False)
    scenarios = [
        _run_scenario(config, scenario, output_dir) for scenario in config.scenarios
    ]
    versions = {}
    for package in ("speculators", "torch", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "valid": all(scenario["valid"] for scenario in scenarios),
        "config": _redacted_config(config),
        "versions": versions,
        "scenarios": scenarios,
    }


def load_config(path: Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate_json(path.read_text(encoding="utf-8"))


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def event_as_dict(event: RequestEvent) -> dict[str, Any]:
    """Expose a stable serialization helper for tests and external tooling."""
    return asdict(event)
