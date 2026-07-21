from __future__ import annotations

import json
import threading
from unittest import mock

import pytest

from speculators.benchmarks.gpu_monitor import (
    GpuDevice,
    GpuMonitor,
    GpuRoleAssignment,
    iter_gpu_samples,
    summarize_gpu_window,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.sampled = threading.Event()
        self.closed = False

    def open(self, gpu_indices):
        return {
            gpu: GpuDevice(gpu=gpu, uuid=f"GPU-{gpu}", name="Fake GPU")
            for gpu in gpu_indices
        }

    def sample(self, gpu):
        self.sampled.set()
        return {
            "utilization_gpu_pct": 75 + gpu,
            "utilization_memory_pct": 40,
            "memory_used_bytes": (8 + gpu) << 30,
            "memory_free_bytes": 40 << 30,
            "memory_total_bytes": 48 << 30,
            "compute_processes": [{"pid": 100 + gpu, "used_memory_bytes": 4 << 30}],
        }

    def close(self):
        self.closed = True


class _BlockingBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.release_sample = threading.Event()

    def sample(self, gpu):
        self.sampled.set()
        self.release_sample.wait(timeout=5)
        return super().sample(gpu)


def test_monitor_streams_nvml_samples_and_finalizes_summary(tmp_path):
    backend = _FakeBackend()
    sample_path = tmp_path / "samples.jsonl"
    summary_path = tmp_path / "summary.json"
    monitor = GpuMonitor(
        [GpuRoleAssignment(0, "producer", "producer")],
        sample_path,
        summary_path,
        poll_seconds=0.01,
        backend=backend,
    )

    monitor.start()
    assert backend.sampled.wait(1)
    summary = monitor.stop()

    assert summary["status"] == "ok"
    assert summary["sample_count_by_gpu"]["0"] >= 1
    assert backend.closed
    assert list(iter_gpu_samples(sample_path))
    assert json.loads(summary_path.read_text())["status"] == "ok"


def test_start_write_failure_closes_backend_and_output(tmp_path, monkeypatch):
    backend = _FakeBackend()
    monitor = GpuMonitor(
        [GpuRoleAssignment(0, "producer", "producer")],
        tmp_path / "samples.jsonl",
        tmp_path / "summary.json",
        backend=backend,
    )
    monkeypatch.setattr(monitor, "_write", mock.Mock(side_effect=OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        monitor.start()

    assert backend.closed
    assert monitor._output is None
    assert not monitor._thread.is_alive()


def test_start_thread_failure_closes_backend_and_output(tmp_path, monkeypatch):
    backend = _FakeBackend()
    monitor = GpuMonitor(
        [GpuRoleAssignment(0, "producer", "producer")],
        tmp_path / "samples.jsonl",
        tmp_path / "summary.json",
        backend=backend,
    )
    monkeypatch.setattr(
        monitor._thread, "start", mock.Mock(side_effect=RuntimeError("no thread"))
    )

    with pytest.raises(RuntimeError, match="no thread"):
        monitor.start()

    assert backend.closed
    assert monitor._output is None
    assert not monitor._thread.is_alive()


def test_stop_timeout_does_not_race_a_blocked_sample(tmp_path, monkeypatch):
    backend = _BlockingBackend()
    monitor = GpuMonitor(
        [GpuRoleAssignment(0, "producer", "producer")],
        tmp_path / "samples.jsonl",
        tmp_path / "summary.json",
        poll_seconds=0.01,
        backend=backend,
    )
    monitor.start()
    assert backend.sampled.wait(1)
    monkeypatch.setattr(monitor._thread, "join", lambda timeout: None)

    summary = monitor.stop()

    assert summary["status"] == "degraded"
    assert summary["sample_count_by_gpu"]["0"] == 0
    assert "GPU monitor thread did not stop" in summary["errors"]
    backend.release_sample.set()
    threading.Thread.join(monitor._thread, timeout=1)
    assert not monitor._thread.is_alive()
    assert backend.closed


def test_stop_write_failure_still_closes_and_writes_degraded_summary(
    tmp_path, monkeypatch
):
    backend = _FakeBackend()
    summary_path = tmp_path / "summary.json"
    monitor = GpuMonitor(
        [GpuRoleAssignment(0, "producer", "producer")],
        tmp_path / "samples.jsonl",
        summary_path,
        poll_seconds=0.01,
        backend=backend,
    )
    monitor.start()
    assert backend.sampled.wait(1)
    original_write = monitor._write

    def fail_session_end(value):
        if value.get("record_type") == "session_end":
            raise OSError("disk full")
        original_write(value)

    monkeypatch.setattr(monitor, "_write", fail_session_end)
    summary = monitor.stop()

    assert summary["status"] == "degraded"
    assert any("session_end write failed" in error for error in summary["errors"])
    assert monitor._output is None
    assert backend.closed
    assert json.loads(summary_path.read_text())["status"] == "degraded"


def test_window_summary_excludes_startup_and_reports_active_time():
    assignment = GpuRoleAssignment(2, "consumer:b4", "consumer:b4")
    samples = [
        {
            "timestamp_monotonic_ns": timestamp,
            "gpu": 2,
            "utilization_gpu_pct": utilization,
            "memory_used_bytes": 10 << 30,
            "compute_pids": [123],
            "compute_processes": [{"pid": 123, "used_memory_bytes": 8 << 30}],
        }
        for timestamp, utilization in (
            (1_000_000_000, 0),
            (2_000_000_000, 80),
            (3_000_000_000, 100),
        )
    ]

    result = summarize_gpu_window(
        samples,
        [assignment],
        start_monotonic_ns=2_000_000_000,
        end_monotonic_ns=3_000_000_000,
    )

    gpu = result["per_gpu"]["2"]
    assert result["valid"]
    assert gpu["sample_count"] == 2
    assert gpu["gpu_utilization_pct"]["mean"] == 90.0
    assert gpu["gpu_utilization_pct"]["p50"] == 80.0
    assert gpu["gpu_utilization_pct"]["p95"] == 100.0
    assert gpu["max_compute_processes"] == 1
    assert gpu["max_memory_used_mib"] == 10 << 10
    assert gpu["max_device_memory_used_mib"] == 10 << 10
    assert gpu["max_compute_process_memory_used_mib"] == 8 << 10
    assert gpu["compute_process_memory_sample_count"] == 2
    assert gpu["compute_process_memory_unavailable_samples"] == 0


def test_window_summary_rejects_one_sample_zero_coverage_and_missing_memory():
    assignment = GpuRoleAssignment(2, "consumer:b4", "consumer:b4")
    sample = {
        "timestamp_monotonic_ns": 2_000_000_000,
        "gpu": 2,
        "utilization_gpu_pct": 80,
        "memory_used_bytes": 10 << 30,
        "compute_pids": [123],
        "compute_processes": [{"pid": 123, "used_memory_bytes": 8 << 30}],
    }
    one = summarize_gpu_window(
        [sample],
        [assignment],
        start_monotonic_ns=1_000_000_000,
        end_monotonic_ns=3_000_000_000,
    )
    assert not one["valid"]
    assert "at least 2" in one["invalid_reasons"][0]

    zero = summarize_gpu_window(
        [sample, dict(sample)],
        [assignment],
        start_monotonic_ns=1_000_000_000,
        end_monotonic_ns=3_000_000_000,
    )
    assert not zero["valid"]
    assert any("zero sample coverage" in reason for reason in zero["invalid_reasons"])

    missing = dict(sample, timestamp_monotonic_ns=3_000_000_000)
    missing["compute_processes"] = [{"pid": 123, "used_memory_bytes": None}]
    incomplete = summarize_gpu_window(
        [sample, missing],
        [assignment],
        start_monotonic_ns=1_000_000_000,
        end_monotonic_ns=3_000_000_000,
    )
    assert not incomplete["valid"]
    assert any(
        "lacks compute-process memory" in reason
        for reason in incomplete["invalid_reasons"]
    )
