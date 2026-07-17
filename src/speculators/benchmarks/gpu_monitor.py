"""Role-aware NVML telemetry for independent-consumer benchmarks."""

from __future__ import annotations

import importlib
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import TextIO


_MIN_COVERAGE_SAMPLES = 2


class GpuMonitorError(RuntimeError):
    """GPU telemetry could not satisfy its benchmark contract."""


@dataclass(frozen=True)
class GpuRoleAssignment:
    gpu: int
    logical_role: str
    process_role: str

    def __post_init__(self) -> None:
        if self.gpu < 0:
            raise ValueError("gpu index must be non-negative")
        if not self.logical_role or not self.process_role:
            raise ValueError("GPU role names must be non-empty")


@dataclass(frozen=True)
class GpuDevice:
    gpu: int
    uuid: str
    name: str


class GpuTelemetryBackend(Protocol):
    def open(self, gpu_indices: Sequence[int]) -> Mapping[int, GpuDevice]: ...

    def sample(self, gpu: int) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _error_text(error: BaseException) -> str:
    message = str(error).replace("\n", " ").strip()
    return f"{type(error).__name__}: {message}"[:512]


class NvmlBackend:
    """Lazy wrapper that never initializes CUDA or Torch."""

    def __init__(self, module: Any = None) -> None:
        self._module = module
        self._handles: dict[int, Any] = {}
        self._opened = False

    def open(self, gpu_indices: Sequence[int]) -> Mapping[int, GpuDevice]:
        if self._opened:
            raise GpuMonitorError("NVML backend is already open")
        if self._module is None:
            try:
                self._module = importlib.import_module("pynvml")
            except ImportError as error:
                raise GpuMonitorError(
                    "NVML monitoring requires nvidia-ml-py; install the nvml extra"
                ) from error
        nvml = self._module
        try:
            nvml.nvmlInit()
            self._opened = True
            devices: dict[int, GpuDevice] = {}
            for gpu in gpu_indices:
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
                self._handles[gpu] = handle
                devices[gpu] = GpuDevice(
                    gpu=gpu,
                    uuid=_as_text(nvml.nvmlDeviceGetUUID(handle)),
                    name=_as_text(nvml.nvmlDeviceGetName(handle)),
                )
            return devices
        except Exception:
            self.close()
            raise

    def sample(self, gpu: int) -> Mapping[str, Any]:
        if gpu not in self._handles:
            raise GpuMonitorError(f"NVML GPU {gpu} is not open")
        nvml = self._module
        handle = self._handles[gpu]
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        processes = []
        for process in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
            used_memory = getattr(process, "usedGpuMemory", None)
            unavailable = getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", None)
            if used_memory == unavailable:
                used_memory = None
            processes.append(
                {"pid": int(process.pid), "used_memory_bytes": used_memory}
            )
        return {
            "utilization_gpu_pct": int(utilization.gpu),
            "utilization_memory_pct": int(utilization.memory),
            "memory_used_bytes": int(memory.used),
            "memory_free_bytes": int(memory.free),
            "memory_total_bytes": int(memory.total),
            "compute_processes": processes,
        }

    def close(self) -> None:
        if not self._opened:
            return
        self._handles.clear()
        self._module.nvmlShutdown()
        self._opened = False


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class GpuMonitor:
    """Stream NVML samples and retain a bounded health summary."""

    def __init__(
        self,
        assignments: Sequence[GpuRoleAssignment],
        sample_path: Path,
        summary_path: Path,
        *,
        poll_seconds: float = 0.5,
        max_compute_processes: int = 1,
        backend: GpuTelemetryBackend | None = None,
    ) -> None:
        self.assignments = tuple(assignments)
        if not self.assignments:
            raise ValueError("at least one GPU role assignment is required")
        if len({value.gpu for value in self.assignments}) != len(self.assignments):
            raise ValueError("GPU role assignments must use distinct GPU indices")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if max_compute_processes < 1:
            raise ValueError("max_compute_processes must be positive")
        self.sample_path = sample_path
        self.summary_path = summary_path
        self.poll_seconds = poll_seconds
        self.max_compute_processes = max_compute_processes
        self.backend = backend or NvmlBackend()
        self._devices: dict[int, GpuDevice] = {}
        self._sample_counts = dict.fromkeys(
            (value.gpu for value in self.assignments), 0
        )
        self._max_processes = dict.fromkeys(
            (value.gpu for value in self.assignments), 0
        )
        self._errors: list[str] = []
        self._violations: list[str] = []
        self._collection_ms: list[float] = []
        self._poll_overruns = 0
        self._collection_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="speculators-nvml-monitor", daemon=True
        )
        self._output: TextIO | None = None
        self._started_at_ns: int | None = None
        self._ended_at_ns: int | None = None
        self._backend_open = False
        self._started = False
        self._stopped = False
        self._summary: dict[str, Any] | None = None

    @property
    def errors(self) -> list[str]:
        with self._collection_lock:
            return list(self._errors)

    @property
    def violations(self) -> list[str]:
        with self._collection_lock:
            return list(self._violations)

    def _close_backend(self) -> None:
        if not self._backend_open:
            return
        try:
            self.backend.close()
        except Exception as error:  # noqa: BLE001
            with self._collection_lock:
                self._errors.append(_error_text(error))
        finally:
            self._backend_open = False

    def _write(self, value: Mapping[str, Any]) -> None:
        output = self._output
        if output is None:
            raise GpuMonitorError("GPU monitor output is not open")
        output.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
        output.write("\n")
        output.flush()

    def start(self) -> None:
        if self._started:
            raise GpuMonitorError("GPU monitor can only be started once")
        self._started = True
        self.sample_path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self.sample_path.open("x", encoding="utf-8", buffering=1)
        try:
            self._devices = dict(
                self.backend.open([value.gpu for value in self.assignments])
            )
            self._backend_open = True
            missing = {value.gpu for value in self.assignments} - set(self._devices)
            if missing:
                raise GpuMonitorError(f"NVML backend omitted GPUs {sorted(missing)}")
        except Exception:
            self._output.close()
            self._output = None
            raise
        self._started_at_ns = time.monotonic_ns()
        self._write(
            {
                "record_type": "session_start",
                "timestamp_monotonic_ns": self._started_at_ns,
                "poll_seconds": self.poll_seconds,
                "assignments": [asdict(value) for value in self.assignments],
                "devices": [asdict(value) for value in self._devices.values()],
            }
        )
        self._thread.start()

    def _run(self) -> None:  # noqa: C901
        try:
            while not self._stop.is_set():
                cycle_started = time.monotonic()
                timestamp_ns = time.monotonic_ns()
                timestamp_unix_ns = time.time_ns()
                for assignment in self.assignments:
                    if self._stop.is_set():
                        return
                    try:
                        metrics = dict(self.backend.sample(assignment.gpu))
                        pids = sorted(
                            int(value["pid"])
                            for value in metrics.get("compute_processes", ())
                        )
                        with self._collection_lock:
                            if self._stop.is_set():
                                return
                            self._sample_counts[assignment.gpu] += 1
                            self._max_processes[assignment.gpu] = max(
                                self._max_processes[assignment.gpu], len(pids)
                            )
                            if len(pids) > self.max_compute_processes:
                                message = (
                                    f"GPU {assignment.gpu} has {len(pids)} compute "
                                    f"processes: {pids}"
                                )
                                if message not in self._violations:
                                    self._violations.append(message)
                            self._write(
                                {
                                    "record_type": "sample",
                                    "timestamp_monotonic_ns": timestamp_ns,
                                    "timestamp_unix_ns": timestamp_unix_ns,
                                    **asdict(assignment),
                                    **metrics,
                                    "compute_pids": pids,
                                }
                            )
                    except Exception as error:  # noqa: BLE001
                        with self._collection_lock:
                            if self._stop.is_set():
                                return
                            message = _error_text(error)
                            if message not in self._errors:
                                self._errors.append(message)
                elapsed_ms = (time.monotonic() - cycle_started) * 1000.0
                with self._collection_lock:
                    if self._stop.is_set():
                        break
                    self._collection_ms.append(elapsed_ms)
                    if elapsed_ms > self.poll_seconds * 1000.0:
                        self._poll_overruns += 1
                self._stop.wait(max(0.0, self.poll_seconds - elapsed_ms / 1000.0))
        finally:
            self._close_backend()

    def stop(self) -> dict[str, Any]:
        if not self._started:
            raise GpuMonitorError("GPU monitor was not started")
        if self._stopped:
            if self._summary is None:
                raise GpuMonitorError("GPU monitor summary is unavailable")
            return self._summary
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.poll_seconds * 4))
        with self._collection_lock:
            if self._thread.is_alive():
                self._errors.append("GPU monitor thread did not stop")
            self._ended_at_ns = time.monotonic_ns()
            if self._output is not None:
                self._write(
                    {
                        "record_type": "session_end",
                        "timestamp_monotonic_ns": self._ended_at_ns,
                    }
                )
                self._output.close()
                self._output = None
            summary = {
                "status": (
                    "ok" if not self._errors and not self._violations else "degraded"
                ),
                "sample_path": str(self.sample_path.resolve()),
                "duration_seconds": (
                    (self._ended_at_ns - self._started_at_ns) / 1e9
                    if self._started_at_ns is not None
                    else None
                ),
                "poll_seconds": self.poll_seconds,
                "poll_overrun_count": self._poll_overruns,
                "collection_latency_ms": {
                    "mean": statistics.fmean(self._collection_ms)
                    if self._collection_ms
                    else None,
                    "max": max(self._collection_ms) if self._collection_ms else None,
                },
                "sample_count_by_gpu": {
                    str(key): value for key, value in self._sample_counts.items()
                },
                "max_compute_processes_by_gpu": {
                    str(key): value for key, value in self._max_processes.items()
                },
                "errors": list(self._errors),
                "ownership_violations": list(self._violations),
            }
            self._summary = summary
            self._stopped = True
        _atomic_write_json(self.summary_path, summary)
        return summary


def iter_gpu_samples(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete samples while tolerating one torn final JSONL line."""

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                if not line.endswith("\n"):
                    return
                raise GpuMonitorError(
                    f"invalid GPU sample JSONL at line {line_number}: {error}"
                ) from error
            if value.get("record_type") == "sample":
                yield value


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
    return float(ordered[index])


def summarize_gpu_window(
    samples: Iterable[Mapping[str, Any]],
    assignments: Sequence[GpuRoleAssignment],
    *,
    start_monotonic_ns: int,
    end_monotonic_ns: int,
    low_utilization_pct: int = 10,
) -> dict[str, Any]:
    """Aggregate NVML active-time samples in one common consumer interval."""

    if end_monotonic_ns <= start_monotonic_ns:
        raise ValueError("GPU measurement window must have positive duration")
    rows: dict[int, list[Mapping[str, Any]]] = {value.gpu: [] for value in assignments}
    for sample in samples:
        timestamp = sample.get("timestamp_monotonic_ns")
        gpu = sample.get("gpu")
        if (
            isinstance(timestamp, int)
            and not isinstance(timestamp, bool)
            and start_monotonic_ns <= timestamp <= end_monotonic_ns
            and gpu in rows
        ):
            rows[gpu].append(sample)
    invalid_reasons = []
    per_gpu = {}
    by_gpu = {value.gpu: value for value in assignments}
    for gpu in sorted(rows):
        gpu_rows = rows[gpu]
        utilization = [
            float(value["utilization_gpu_pct"])
            for value in gpu_rows
            if value.get("utilization_gpu_pct") is not None
        ]
        timestamps = [int(value["timestamp_monotonic_ns"]) for value in gpu_rows]
        process_counts = [len(value.get("compute_pids", ())) for value in gpu_rows]
        memory = [
            int(value["memory_used_bytes"])
            for value in gpu_rows
            if value.get("memory_used_bytes") is not None
        ]
        if not gpu_rows:
            invalid_reasons.append(f"GPU {gpu} has no samples in the steady window")
        if gpu_rows and not any(process_counts):
            invalid_reasons.append(
                f"GPU {gpu} has no compute process in the steady window"
            )
        per_gpu[str(gpu)] = {
            **asdict(by_gpu[gpu]),
            "sample_count": len(gpu_rows),
            "coverage_seconds": (
                (max(timestamps) - min(timestamps)) / 1e9
                if len(timestamps) >= _MIN_COVERAGE_SAMPLES
                else 0.0
            ),
            "gpu_utilization_pct": {
                "mean": statistics.fmean(utilization) if utilization else None,
                "p50": _percentile(utilization, 0.50),
                "p95": _percentile(utilization, 0.95),
                "low_fraction": (
                    sum(value < low_utilization_pct for value in utilization)
                    / len(utilization)
                    if utilization
                    else None
                ),
            },
            "max_memory_used_mib": max(memory) / (1 << 20) if memory else None,
            "max_compute_processes": max(process_counts, default=0),
        }
    return {
        "interval": "common_consumer_steady_overlap",
        "start_monotonic_ns": start_monotonic_ns,
        "end_monotonic_ns": end_monotonic_ns,
        "duration_seconds": (end_monotonic_ns - start_monotonic_ns) / 1e9,
        "low_utilization_threshold_pct": low_utilization_pct,
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "per_gpu": per_gpu,
    }
