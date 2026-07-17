from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import speculators.data_generation.artifact_cache as artifact_cache_module
from speculators.data_generation.artifact_cache import (
    ArtifactLockTimeoutError,
    HiddenStateArtifactCache,
    canonical_hidden_state_extraction_namespace,
    canonical_hidden_state_request_id,
)
from speculators.data_generation.offline import check_hidden_states


def _tensors(tokens: tuple[int, ...] = (1, 2, 3)) -> dict[str, torch.Tensor]:
    return {
        "token_ids": torch.tensor(tokens),
        "hidden_states": torch.arange(len(tokens) * 4, dtype=torch.float32).reshape(
            len(tokens), 4
        ),
    }


def _validate(data: dict[str, torch.Tensor]) -> None:
    check_hidden_states(data, [1, 2, 3])


def _request_id() -> str:
    return canonical_hidden_state_request_id("model", {"input_ids": [1, 2, 3]})


def _spawn_cache_worker(root, ready, start, generation_count, results):
    cache = HiddenStateArtifactCache(
        root,
        artifact_ttl_seconds=None,
        lock_timeout_seconds=10,
        lock_poll_seconds=0.01,
    )
    ready.put(True)
    start.wait(10)

    def create():
        with generation_count.get_lock():
            generation_count.value += 1
        time.sleep(0.4)
        return _tensors()

    result = cache.get_or_create(_request_id(), create, _validate)
    results.put((result.cache_hit, result.data["token_ids"].tolist()))


def test_canonical_request_id_is_stable_and_covers_semantics():
    first = canonical_hidden_state_request_id(
        "model",
        {
            "input_ids": [1, 2, 3],
            "messages": [{"role": "user", "content": {"text": "hello", "x": 1}}],
        },
        namespace="layers:2,18,33",
    )
    reordered = canonical_hidden_state_request_id(
        "model",
        {
            "messages": [{"content": {"x": 1, "text": "hello"}, "role": "user"}],
            "input_ids": [1, 2, 3],
        },
        namespace="layers:2,18,33",
    )

    assert first == reordered
    assert first != canonical_hidden_state_request_id(
        "model", {"input_ids": [1, 2, 4]}, namespace="layers:2,18,33"
    )
    assert first != canonical_hidden_state_request_id(
        "model", {"input_ids": [1, 2, 3]}, namespace="layers:2,18,36"
    )
    assert first != canonical_hidden_state_request_id(
        "other-model", {"input_ids": [1, 2, 3]}, namespace="layers:2,18,33"
    )


def test_extraction_namespace_fingerprints_layers_and_user_namespace():
    first = canonical_hidden_state_extraction_namespace(
        (2, 18, 33), user_namespace="revision-a"
    )
    assert first == canonical_hidden_state_extraction_namespace(
        (2, 18, 33), user_namespace="revision-a"
    )
    assert first != canonical_hidden_state_extraction_namespace(
        (2, 18, 36), user_namespace="revision-a"
    )
    assert first != canonical_hidden_state_extraction_namespace(
        (2, 18, 33), user_namespace="revision-b"
    )


@pytest.mark.parametrize("tokens", [[], [1, True], [[1, 2]]])
def test_canonical_request_id_rejects_invalid_tokens(tokens):
    with pytest.raises(ValueError, match="input_ids"):
        canonical_hidden_state_request_id("model", {"input_ids": tokens})


def test_sequential_request_publishes_once_then_hits(tmp_path):
    cache = HiddenStateArtifactCache(tmp_path, artifact_ttl_seconds=None)
    generations = 0

    def create():
        nonlocal generations
        generations += 1
        return _tensors()

    first = cache.get_or_create(_request_id(), create, _validate)
    second = cache.get_or_create(_request_id(), create, _validate)

    assert generations == 1
    assert not first.cache_hit
    assert second.cache_hit
    assert torch.equal(first.data["hidden_states"], second.data["hidden_states"])
    assert load_file(first.path)["token_ids"].tolist() == [1, 2, 3]
    assert cache.snapshot_stats() == {
        "schema_version": 1,
        "logical_requests": 2,
        "hits": 1,
        "misses": 1,
        "coalesced_waiters": 0,
        "retry_generations": 0,
        "publishes": 1,
        "generation_failures": 0,
        "publish_failures": 0,
        "invalid_artifacts_removed": 0,
        "expired_artifacts_removed": 0,
        "stale_temps_removed": 0,
        "lock_timeouts": 0,
    }


def test_independent_processes_coalesce_one_generation(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    generation_count = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_spawn_cache_worker,
            args=(str(tmp_path), ready, start, generation_count, results),
        )
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=20)
        start.set()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=5) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert generation_count.value == 1
    assert sum(cache_hit for cache_hit, _tokens in outcomes) == 2
    assert all(tokens == [1, 2, 3] for _cache_hit, tokens in outcomes)
    stats = HiddenStateArtifactCache(
        tmp_path, artifact_ttl_seconds=None
    ).snapshot_stats()
    assert stats["logical_requests"] == 3
    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["publishes"] == 1
    assert stats["coalesced_waiters"] >= 1


def test_failed_owner_allows_waiter_to_retry(tmp_path):
    cache = HiddenStateArtifactCache(
        tmp_path,
        artifact_ttl_seconds=None,
        lock_timeout_seconds=5,
        lock_poll_seconds=0.01,
    )
    start = threading.Barrier(2)
    owner_entered = threading.Event()
    release_owner = threading.Event()
    attempt_lock = threading.Lock()
    attempts = 0

    def create():
        nonlocal attempts
        with attempt_lock:
            attempts += 1
            attempt = attempts
        if attempt == 1:
            owner_entered.set()
            assert release_owner.wait(5)
            raise RuntimeError("producer failed")
        return _tensors()

    def request():
        start.wait()
        try:
            result = cache.get_or_create(_request_id(), create, _validate)
        except RuntimeError:
            return "failed"
        return "hit" if result.cache_hit else "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request) for _ in range(2)]
        assert owner_entered.wait(5)
        time.sleep(0.1)
        release_owner.set()
        outcomes = [future.result(timeout=10) for future in futures]

    assert sorted(outcomes) == ["failed", "published"]
    assert attempts == 2
    assert cache.artifact_path(_request_id()).is_file()
    stats = cache.snapshot_stats()
    assert stats["generation_failures"] == 1
    assert stats["publishes"] == 1
    assert stats["retry_generations"] == 1


def test_partial_publish_is_never_visible_and_is_retried(tmp_path, monkeypatch):
    cache = HiddenStateArtifactCache(
        tmp_path, artifact_ttl_seconds=None, stale_temp_seconds=0.01
    )
    original_publish = cache._publish

    def fail_publish(request_id, _data, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.parent / f".{request_id}.dead.tmp"
        partial.write_bytes(b"partial")
        raise OSError("disk write failed")

    monkeypatch.setattr(cache, "_publish", fail_publish)
    with pytest.raises(OSError, match="disk write failed"):
        cache.get_or_create(_request_id(), _tensors, _validate)
    assert not cache.artifact_path(_request_id()).exists()

    partial = next(cache.artifact_path(_request_id()).parent.glob(".*.tmp"))
    old = time.time() - 1
    os.utime(partial, (old, old))
    monkeypatch.setattr(cache, "_publish", original_publish)
    result = cache.get_or_create(_request_id(), _tensors, _validate)

    assert not result.cache_hit
    assert result.path.is_file()
    assert not partial.exists()
    stats = cache.snapshot_stats()
    assert stats["publish_failures"] == 1
    assert stats["stale_temps_removed"] == 1


def test_atomic_name_is_absent_while_temporary_file_is_written(tmp_path, monkeypatch):
    cache = HiddenStateArtifactCache(tmp_path, artifact_ttl_seconds=None)
    write_started = threading.Event()
    finish_write = threading.Event()
    original_save = save_file

    def paused_save(data, path):
        Path(path).write_bytes(b"partial")
        write_started.set()
        assert finish_write.wait(5)
        original_save(data, path)

    monkeypatch.setattr(artifact_cache_module, "save_file", paused_save)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            cache.get_or_create, _request_id(), _tensors, _validate
        )
        assert write_started.wait(5)
        assert not cache.artifact_path(_request_id()).exists()
        finish_write.set()
        result = future.result(timeout=10)

    assert result.path.is_file()
    assert load_file(result.path)["token_ids"].tolist() == [1, 2, 3]


def test_corrupt_artifact_is_removed_before_regeneration(tmp_path):
    cache = HiddenStateArtifactCache(tmp_path, artifact_ttl_seconds=None)
    target = cache.artifact_path(_request_id())
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not safetensors")

    result = cache.get_or_create(_request_id(), _tensors, _validate)

    assert not result.cache_hit
    assert load_file(target)["token_ids"].tolist() == [1, 2, 3]
    assert cache.snapshot_stats()["invalid_artifacts_removed"] == 1


def test_cleanup_removes_expired_artifact_and_stale_temp(tmp_path):
    cache = HiddenStateArtifactCache(
        tmp_path, artifact_ttl_seconds=1, stale_temp_seconds=1
    )
    result = cache.get_or_create(_request_id(), _tensors, _validate)
    stale_temp = result.path.parent / f".{_request_id()}.dead.tmp"
    stale_temp.write_bytes(b"partial")
    stale_stats_temp = tmp_path / ".stats.123.dead.tmp"
    stale_stats_temp.write_bytes(b"partial")
    old = time.time() - 10
    os.utime(result.path, (old, old))
    os.utime(stale_temp, (old, old))
    os.utime(stale_stats_temp, (old, old))

    cleanup = cache.cleanup_stale(now=time.time())

    assert cleanup == {
        "expired_artifacts_removed": 1,
        "stale_temps_removed": 2,
    }
    assert not result.path.exists()
    assert not stale_temp.exists()
    assert not stale_stats_temp.exists()
    stats = cache.snapshot_stats()
    assert stats["expired_artifacts_removed"] == 1
    assert stats["stale_temps_removed"] == 2


def test_lock_timeout_is_counted(tmp_path):
    owner = HiddenStateArtifactCache(
        tmp_path, artifact_ttl_seconds=None, lock_timeout_seconds=5
    )
    waiter = HiddenStateArtifactCache(
        tmp_path,
        artifact_ttl_seconds=None,
        lock_timeout_seconds=0.05,
        lock_poll_seconds=0.01,
    )
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with owner._request_lock(_request_id()):
            locked.set()
            assert release.wait(5)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert locked.wait(5)
    try:
        with pytest.raises(ArtifactLockTimeoutError):
            waiter.get_or_create(_request_id(), _tensors, _validate)
    finally:
        release.set()
        thread.join(timeout=5)

    stats = waiter.snapshot_stats()
    assert stats["logical_requests"] == 1
    assert stats["lock_timeouts"] == 1
    assert not waiter.artifact_path(_request_id()).exists()
