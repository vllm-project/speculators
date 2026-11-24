"""Lock-wait behaviour for hidden-states files written by the vLLM server.

The server holds an exclusive flock on "<path>.lock" from request scheduling
until its writer thread finishes save_file; wait_for_lock / wait_for_lock_async
are the client's completeness proof before it publishes the file. These tests
exercise the wait against real flocks: delayed writer completion, a writer that
never finishes, and stale lock files nobody holds.
"""

import asyncio
import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from speculators.data_generation import vllm_client
from speculators.data_generation.vllm_client import wait_for_lock, wait_for_lock_async


@pytest.fixture
def lock_path(tmp_path):
    # The writer creates the data file only once save_file starts, so the lock
    # typically exists on its own for most of the wait.
    path = tmp_path / "hs.safetensors.lock"
    path.touch()
    return path


def _hold_lock(lock_path: Path, hold_seconds: float) -> threading.Timer:
    """Hold LOCK_EX on the lock file, like the server's writer thread does."""
    fd = os.open(lock_path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    timer = threading.Timer(hold_seconds, os.close, args=[fd])
    timer.start()
    return timer


def test_unheld_lock_is_acquired_and_removed(lock_path):
    wait_for_lock(str(lock_path))

    assert not lock_path.exists()


def test_delayed_writer_completion_is_awaited(lock_path):
    timer = _hold_lock(lock_path, hold_seconds=0.5)

    start = time.monotonic()
    wait_for_lock(str(lock_path), timeout=10.0)
    elapsed = time.monotonic() - start

    timer.join()
    assert elapsed >= 0.5
    assert not lock_path.exists()


def test_writer_failure_times_out_with_diagnostics(lock_path):
    lock_path.with_suffix("").write_bytes(b"\0" * 1024)
    # The lock stays held: a stuck writer (or a dead process on a filesystem
    # that does not release flocks) must surface as a loud, bounded failure.
    timer = _hold_lock(lock_path, hold_seconds=60.0)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError) as excinfo:
            wait_for_lock(
                str(lock_path),
                timeout=0.5,
                poll_interval=0.05,
                context="sample 2969",
            )
        elapsed = time.monotonic() - start
    finally:
        timer.cancel()
        timer.join()

    # The wait respects the configured bound instead of hanging forever.
    assert elapsed < 5.0
    message = str(excinfo.value)
    assert str(lock_path) in message
    assert "limit 0.5s" in message
    assert "sample 2969" in message
    assert "data file size" in message


def test_timeout_reports_missing_data_file(lock_path):
    timer = _hold_lock(lock_path, hold_seconds=60.0)
    try:
        with pytest.raises(TimeoutError, match="data file not yet created"):
            wait_for_lock(str(lock_path), timeout=0.2, poll_interval=0.05)
    finally:
        timer.cancel()
        timer.join()


def test_long_wait_logs_progress(lock_path, caplog, monkeypatch):
    monkeypatch.setattr(vllm_client, "_LOCK_LOG_INTERVAL", 0.2)
    timer = _hold_lock(lock_path, hold_seconds=0.6)
    try:
        with caplog.at_level("WARNING"):
            wait_for_lock(str(lock_path), timeout=10.0, poll_interval=0.05)
    finally:
        timer.join()

    progress = [r for r in caplog.records if "Still waiting" in r.message]
    assert progress, "expected progress logs while the writer holds the lock"
    assert all(str(lock_path) in r.message for r in progress)


def test_async_delayed_writer_completion(lock_path):
    timer = _hold_lock(lock_path, hold_seconds=0.3)

    async def scenario():
        await wait_for_lock_async(str(lock_path), timeout=10.0, poll_interval=0.05)

    asyncio.run(scenario())
    timer.join()
    assert not lock_path.exists()


def test_async_writer_failure_times_out_with_diagnostics(lock_path):
    timer = _hold_lock(lock_path, hold_seconds=60.0)

    async def scenario():
        await wait_for_lock_async(
            str(lock_path), timeout=0.2, poll_interval=0.05, context="sample 7"
        )

    try:
        with pytest.raises(TimeoutError) as excinfo:
            asyncio.run(scenario())
    finally:
        timer.cancel()
        timer.join()

    message = str(excinfo.value)
    assert str(lock_path) in message
    assert "limit 0.2s" in message
    assert "sample 7" in message
