from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, cast

import hs_connectors.transfer as transfer_module
import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file

import speculators.train.data as data_module
import speculators.train.dataloader as dataloader_module
from hs_connectors import FileTransfer
from speculators.data_generation.windowed_artifacts import (
    ArtifactPriority,
    GenerationClaim,
    WindowedArtifactCoordinator,
    WindowedArtifactError,
)
from speculators.train.data import WINDOWED_LEASE_KEY, ArrowDataset
from speculators.train.dataloader import WindowedBatchSampler

if TYPE_CHECKING:
    from pathlib import Path


def _write_dataset(path: Path) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[0, 1, 1]],
            "seq_len": [3],
        }
    ).with_format("torch")
    dataset.save_to_disk(path)


def _write_multi_dataset(path: Path, count: int = 6) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[index + 1, index + 2, index + 3] for index in range(count)],
            "loss_mask": [[0, 1, 1] for _ in range(count)],
            "seq_len": [3 for _ in range(count)],
        }
    ).with_format("torch")
    dataset.save_to_disk(path)


def _hidden_states() -> dict[str, torch.Tensor]:
    return {
        "token_ids": torch.tensor([1, 2, 3]),
        "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }


def _arrow_dataset(
    data_path: Path,
    *,
    shared_path: Path | None,
    hidden_states_path: Path,
    on_generate: Literal["cache", "delete"] = "delete",
) -> ArrowDataset:
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(hidden_states_path),
        model="model",
        on_missing="generate",
        on_generate=on_generate,
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace=("layers:2,18,33" if shared_path else None),
        shared_artifacts_ttl_seconds=None,
    )
    dataset.client = cast("Any", object())
    return dataset


class _SingleBatchSampler:
    epoch = 0
    seed = 0
    rank = 0
    num_replicas = 1
    batch_max_length = 128
    lengths = [3]

    @staticmethod
    def _generate_batches(_epoch: int) -> list[list[int]]:
        return [[0]]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _SequentialSampler:
    epoch = 0
    seed = 0
    rank = 0
    num_replicas = 1
    batch_max_length = 128

    def __init__(self, count: int) -> None:
        self.lengths = [3] * count

    def _generate_batches(self, _epoch: int) -> list[list[int]]:
        return [[index] for index in range(len(self.lengths))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_shared_dataset_requires_identity_namespace(tmp_path):
    data_path = tmp_path / "data"
    _write_dataset(data_path)

    with pytest.raises(ValueError, match="shared_artifacts_namespace is required"):
        ArrowDataset(
            max_len=128,
            datapath=data_path,
            model="model",
            shared_artifacts_path=tmp_path / "shared",
        )


def test_windowed_dataset_requires_online_generation(tmp_path):
    data_path = tmp_path / "data"
    _write_dataset(data_path)

    with pytest.raises(ValueError, match="require on_missing='generate'"):
        ArrowDataset(
            max_len=128,
            datapath=data_path,
            model="model",
            on_missing="raise",
            shared_artifacts_path=tmp_path / "shared",
            shared_artifacts_namespace="layers:2,18,33",
            shared_artifacts_consumer_id="consumer",
        )


def test_windowed_dataset_derives_and_overrides_artifact_timeouts(tmp_path):
    data_path = tmp_path / "data"
    _write_dataset(data_path)

    defaults = ArrowDataset(max_len=128, datapath=data_path, model="model")
    derived = ArrowDataset(
        max_len=128,
        datapath=data_path,
        model="model",
        request_timeout=10,
        max_retries=2,
    )
    explicit = ArrowDataset(
        max_len=128,
        datapath=data_path,
        model="model",
        request_timeout=10,
        max_retries=2,
        shared_artifacts_acquire_timeout_seconds=73,
        shared_artifacts_lease_timeout_seconds=91,
    )

    assert defaults.shared_artifacts_acquire_timeout_seconds == 499
    assert derived.shared_artifacts_acquire_timeout_seconds == 41
    assert derived.shared_artifacts_lease_timeout_seconds == 3600
    assert explicit.shared_artifacts_acquire_timeout_seconds == 73
    assert explicit.shared_artifacts_lease_timeout_seconds == 91


def _successful_generator(service_path: Path, calls: list[Path]):
    def generate(*_args, **_kwargs):
        path = service_path / f"request-{len(calls)}.safetensors"
        save_file(_hidden_states(), path)
        (path.parent / f"{path.name}.lock").touch()
        calls.append(path)
        return str(path)

    return generate


def test_shared_dataset_requests_publish_once_and_delete_service_temporary(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        _successful_generator(service_path, calls),
    )
    first = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "first-index",
    )
    second = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "second-index",
    )

    first_data = first._maybe_generate_hs(0)
    second_data = second._maybe_generate_hs(0)

    assert first_data is not None
    assert second_data is not None
    assert torch.equal(first_data["hidden_states"], second_data["hidden_states"])
    assert len(calls) == 1
    assert not calls[0].exists()
    assert not (calls[0].parent / f"{calls[0].name}.lock").exists()
    assert first.artifact_cache is not None
    stats = first.artifact_cache.snapshot_stats()
    assert stats["logical_requests"] == 2
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["publishes"] == 1


def test_windowed_dataset_dispatches_reads_acks_and_cleans_final_window(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_dataset(data_path)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        request_timeout=2,
    )
    dataset.client = cast("Any", object())
    monkeypatch.setattr(
        dataset,
        "_materialize_shared_hs",
        lambda _index, _dataset_item, _client_item: _hidden_states(),
    )
    base_sampler = _SingleBatchSampler()
    stream_id = dataset.configure_windowed_stream(base_sampler)
    sampler = WindowedBatchSampler(
        base_sampler,
        stream_id=stream_id,
        request_id_for_index=dataset.windowed_request_id,
    )
    samples = sampler.full_epoch_samples(0)
    dataset.prepare_windowed_epoch(samples, cursor=0, reset=True)
    dataset.start_windowed_producer()

    item = dataset[samples[0]]
    assert item is not None
    lease = item.pop(WINDOWED_LEASE_KEY)
    assert dataset.ack_windowed_batch([lease]) == 1
    dataset.stop_windowed_producer(completed=True)

    with WindowedArtifactCoordinator(shared_path) as coordinator:
        snapshot = coordinator.snapshot()
    assert snapshot["retained_artifacts"] == 0
    assert snapshot["consumers"][0]["state"] == "completed"
    assert dataset.artifact_cache is not None
    stats = dataset.artifact_cache.snapshot_stats()
    assert stats["logical_requests"] == 1
    assert stats["publishes"] == 1


def test_windowed_dataset_invalidates_and_regenerates_corrupt_ready_artifact(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_dataset(data_path)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        request_timeout=2,
    )
    dataset.client = cast("Any", object())
    monkeypatch.setattr(
        dataset,
        "_materialize_shared_hs",
        lambda _index, _dataset_item, _client_item: _hidden_states(),
    )
    base_sampler = _SingleBatchSampler()
    stream_id = dataset.configure_windowed_stream(base_sampler)
    sampler = WindowedBatchSampler(
        base_sampler,
        stream_id=stream_id,
        request_id_for_index=dataset.windowed_request_id,
    )
    samples = sampler.full_epoch_samples(0)
    dataset.prepare_windowed_epoch(samples, cursor=0, reset=True)
    assert dataset.artifact_cache is not None
    corrupt_path = dataset.artifact_cache.artifact_path(samples[0].request_id)
    corrupt_path.parent.mkdir(parents=True)
    save_file(
        {
            "token_ids": torch.tensor([4, 5, 6]),
            "hidden_states": torch.zeros(3, 4),
        },
        corrupt_path,
    )
    with WindowedArtifactCoordinator(shared_path) as coordinator:
        claim = coordinator.claim_generation("initial", stream_id=stream_id)[0]
        coordinator.complete_generation(
            "initial",
            claim,
            path=corrupt_path,
            size_bytes=corrupt_path.stat().st_size,
        )

    dataset.start_windowed_producer()
    try:
        item = dataset[samples[0]]
        assert item is not None
        assert item["input_ids"].tolist() == [1, 2, 3]
        lease = item.pop(WINDOWED_LEASE_KEY)
        assert dataset.ack_windowed_batch([lease]) == 1
    finally:
        dataset.stop_windowed_producer(completed=True)

    stats = dataset.artifact_cache.snapshot_stats()
    assert stats["invalid_artifacts_removed"] == 1
    assert stats["publishes"] == 1


def test_windowed_producer_cleans_abandoned_cache_temporary(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_dataset(data_path)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        request_timeout=2,
    )
    dataset.client = cast("Any", object())
    monkeypatch.setattr(
        dataset,
        "_materialize_shared_hs",
        lambda _index, _dataset_item, _client_item: _hidden_states(),
    )
    base_sampler = _SingleBatchSampler()
    stream_id = dataset.configure_windowed_stream(base_sampler)
    sampler = WindowedBatchSampler(
        base_sampler,
        stream_id=stream_id,
        request_id_for_index=dataset.windowed_request_id,
    )
    dataset.prepare_windowed_epoch(sampler.full_epoch_samples(0), cursor=0, reset=True)
    assert dataset.artifact_cache is not None
    temporary = (
        dataset.artifact_cache.artifact_path("a" * 64).parent
        / f".{('a' * 64)}.dead.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(b"partial")
    old = time.time() - dataset.artifact_cache.stale_temp_seconds - 1
    temporary.touch()
    data_module.os.utime(temporary, (old, old))

    dataset.start_windowed_producer()
    try:
        deadline = time.monotonic() + 2
        while temporary.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("stale cache temporary was not cleaned")
            time.sleep(0.01)
    finally:
        dataset.stop_windowed_producer()

    assert dataset.artifact_cache.snapshot_stats()["stale_temps_removed"] == 1


@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_windowed_scheduling_is_independent_of_dataloader_workers(
    tmp_path, monkeypatch, num_workers
):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_multi_dataset(data_path)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        request_timeout=5,
    )
    dataset.client = cast("Any", object())

    def materialize(_index, dataset_item, _client_item):
        tokens = dataset_item["input_ids"]
        return {
            "token_ids": tokens,
            "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        }

    monkeypatch.setattr(dataset, "_materialize_shared_hs", materialize)
    loader = dataloader_module._setup_dataloader(
        dataset,
        total_seq_len=6,
        hidden_size=1,
        num_workers=num_workers,
        num_target_layers=3,
        prefetch_factor=2,
    )
    sampler = loader.batch_sampler
    assert isinstance(sampler, WindowedBatchSampler)
    sampler.set_epoch(0)
    samples = sampler.full_epoch_samples(0)
    dataset.prepare_windowed_epoch(samples, cursor=0, reset=True)
    iterator = iter(loader)
    dataset.start_windowed_producer()

    seen_sequences: list[int] = []
    for batch in iterator:
        leases = batch.pop(data_module.WINDOWED_BATCH_LEASES_KEY)
        seen_sequences.extend(lease["sequence"] for lease in leases)
        dataset.ack_windowed_batch(leases)
    dataset.stop_windowed_producer(completed=True)

    assert seen_sequences == list(range(len(samples)))
    with WindowedArtifactCoordinator(shared_path) as coordinator:
        snapshot = coordinator.snapshot()
    assert snapshot["retained_artifacts"] == 0
    assert snapshot["consumers"][0]["cursor"] == len(samples)


def test_windowed_producer_runs_bounded_concurrent_capture_batches(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_multi_dataset(data_path, count=8)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        shared_artifacts_lookahead=7,
        shared_artifacts_max_prefetch_per_consumer=8,
        shared_artifacts_capture_batch_size=4,
        shared_artifacts_capture_batch_wait_seconds=0,
        request_timeout=5,
    )
    dataset.client = cast("Any", object())
    lock = threading.Lock()
    active = 0
    peak = 0

    def materialize(_index, dataset_item, _client_item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return {
                "token_ids": dataset_item["input_ids"],
                "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(dataset, "_materialize_shared_hs", materialize)
    base_sampler = _SequentialSampler(8)
    stream_id = dataset.configure_windowed_stream(base_sampler)
    sampler = WindowedBatchSampler(
        base_sampler,
        stream_id=stream_id,
        request_id_for_index=dataset.windowed_request_id,
    )
    dataset.prepare_windowed_epoch(sampler.full_epoch_samples(0), cursor=0, reset=True)

    dataset.start_windowed_producer()
    try:
        deadline = time.monotonic() + 5
        with WindowedArtifactCoordinator(shared_path) as coordinator:
            while coordinator.snapshot()["artifact_states"].get("ready", 0) != 8:
                if time.monotonic() >= deadline:
                    raise TimeoutError("concurrent capture batch did not complete")
                time.sleep(0.01)
    finally:
        dataset.stop_windowed_producer()

    assert peak == 4


def test_windowed_producer_stop_releases_blocked_claim(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_multi_dataset(data_path, count=1)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        shared_artifacts_lookahead=0,
        shared_artifacts_max_prefetch_per_consumer=1,
        shared_artifacts_capture_batch_size=1,
        shared_artifacts_capture_batch_wait_seconds=0,
        shared_artifacts_claim_timeout_seconds=0.3,
        request_timeout=5,
    )
    dataset.client = cast("Any", object())
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def materialize(_index, dataset_item, _client_item):
        entered.set()
        try:
            release.wait(timeout=5)
            return {
                "token_ids": dataset_item["input_ids"],
                "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            }
        finally:
            finished.set()

    monkeypatch.setattr(dataset, "_materialize_shared_hs", materialize)
    base_sampler = _SequentialSampler(1)
    stream_id = dataset.configure_windowed_stream(base_sampler)
    sampler = WindowedBatchSampler(
        base_sampler,
        stream_id=stream_id,
        request_id_for_index=dataset.windowed_request_id,
    )
    dataset.prepare_windowed_epoch(sampler.full_epoch_samples(0), cursor=0, reset=True)

    dataset.start_windowed_producer()
    try:
        assert entered.wait(timeout=2)
        started = time.monotonic()
        dataset.stop_windowed_producer()
        assert time.monotonic() - started < 2
        with WindowedArtifactCoordinator(shared_path) as coordinator:
            assert coordinator.snapshot()["artifact_states"] == {"queued": 1}
    finally:
        release.set()
        assert finished.wait(timeout=2)


def test_windowed_capture_batch_isolates_one_failed_claim(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    shared_path = tmp_path / "shared"
    _write_multi_dataset(data_path, count=4)
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        transfer=FileTransfer(tmp_path / "index"),
        model="model",
        on_missing="generate",
        on_generate="delete",
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer",
        shared_artifacts_lookahead=3,
        shared_artifacts_max_prefetch_per_consumer=4,
        shared_artifacts_capture_batch_size=4,
        shared_artifacts_capture_batch_wait_seconds=0,
        shared_artifacts_generation_attempts=1,
        request_timeout=5,
    )
    dataset.client = cast("Any", object())

    def materialize(index, dataset_item, _client_item):
        if index == 1:
            raise RuntimeError("isolated capture failure")
        return {
            "token_ids": dataset_item["input_ids"],
            "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        }

    monkeypatch.setattr(dataset, "_materialize_shared_hs", materialize)
    base_sampler = _SequentialSampler(4)
    stream_id = dataset.configure_windowed_stream(base_sampler)
    sampler = WindowedBatchSampler(
        base_sampler,
        stream_id=stream_id,
        request_id_for_index=dataset.windowed_request_id,
    )
    dataset.prepare_windowed_epoch(sampler.full_epoch_samples(0), cursor=0, reset=True)

    dataset.start_windowed_producer()
    try:
        deadline = time.monotonic() + 5
        with WindowedArtifactCoordinator(shared_path) as coordinator:
            while True:
                states = coordinator.snapshot()["artifact_states"]
                if states.get("ready", 0) == 3 and states.get("failed", 0) == 1:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "mixed capture batch did not reach terminal state"
                    )
                time.sleep(0.01)
    finally:
        dataset.stop_windowed_producer()


@pytest.mark.parametrize("stale_operation", ["complete", "fail"])
def test_windowed_capture_batch_isolates_stale_terminal_result(
    tmp_path, monkeypatch, stale_operation
):
    dataset = cast("Any", ArrowDataset.__new__(ArrowDataset))
    dataset._windowed_producer_stop = threading.Event()
    dataset.shared_artifacts_consumer_id = "consumer"
    dataset.shared_artifacts_claim_timeout_seconds = 0.03
    claims = (
        GenerationClaim(
            request_id="a" * 64,
            stream_id="1" * 64,
            dataset_index=0,
            generation=1,
            priority=ArtifactPriority.DEMAND,
        ),
        GenerationClaim(
            request_id="b" * 64,
            stream_id="1" * 64,
            dataset_index=1,
            generation=1,
            priority=ArtifactPriority.DEMAND,
        ),
    )
    completed = []

    def produce(_cache, claim):
        if stale_operation == "fail" and claim is claims[0]:
            raise RuntimeError("capture failed")
        return tmp_path / f"{claim.request_id}.safetensors", 1

    class _Coordinator:
        @staticmethod
        def heartbeat(_consumer_id):
            return None

        @staticmethod
        def complete_generation(_owner, claim, **_kwargs):
            if stale_operation == "complete" and claim is claims[0]:
                raise WindowedArtifactError("stale completion")
            completed.append(claim)

        @staticmethod
        def fail_generation(_owner, claim, _error):
            if stale_operation == "fail" and claim is claims[0]:
                raise WindowedArtifactError("stale failure")

        @staticmethod
        def renew_generation_claims(_owner, _claims):
            return None

        @staticmethod
        def release_generation_claims(_owner, _claims):
            return None

    monkeypatch.setattr(dataset, "_produce_windowed_claim", produce)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dataset._run_windowed_claim_batch(
            _Coordinator(), cast("Any", object()), executor, "owner", claims
        )

    assert claims[1] in completed


def test_windowed_capture_batch_drops_only_stale_renewal(monkeypatch):
    dataset = cast("Any", ArrowDataset.__new__(ArrowDataset))
    dataset._windowed_producer_stop = threading.Event()
    dataset.shared_artifacts_consumer_id = "consumer"
    dataset.shared_artifacts_claim_timeout_seconds = 0.03
    release = threading.Event()
    claims = (
        GenerationClaim("a" * 64, "1" * 64, 0, 1, ArtifactPriority.DEMAND),
        GenerationClaim("b" * 64, "1" * 64, 1, 1, ArtifactPriority.DEMAND),
    )
    completed = []

    def produce(_cache, claim):
        assert release.wait(2)
        return cast("Any", object()), claim.dataset_index

    class _Coordinator:
        @staticmethod
        def heartbeat(_consumer_id):
            return None

        @staticmethod
        def complete_generation(_owner, claim, **_kwargs):
            completed.append(claim)

        @staticmethod
        def fail_generation(_owner, _claim, _error):
            raise AssertionError("generation should not fail")

        @staticmethod
        def renew_generation_claims(_owner, renewed):
            if renewed[0] is claims[0]:
                raise WindowedArtifactError("stale renewal")
            release.set()

        @staticmethod
        def release_generation_claims(_owner, _claims):
            return None

    monkeypatch.setattr(dataset, "_produce_windowed_claim", produce)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dataset._run_windowed_claim_batch(
            _Coordinator(), cast("Any", object()), executor, "owner", claims
        )

    assert completed == [claims[1]]


def test_unconfigured_dataset_keeps_existing_per_request_delete_behavior(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        _successful_generator(service_path, calls),
    )
    first = _arrow_dataset(
        data_path,
        shared_path=None,
        hidden_states_path=tmp_path / "first-index",
    )
    second = _arrow_dataset(
        data_path,
        shared_path=None,
        hidden_states_path=tmp_path / "second-index",
    )

    assert first._maybe_generate_hs(0) is not None
    assert second._maybe_generate_hs(0) is not None

    assert first.artifact_cache is None
    assert len(calls) == 2
    assert all(not path.exists() for path in calls)


def test_shared_dataset_does_not_publish_partial_service_artifact(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []

    def generate(*_args, **_kwargs):
        path = service_path / f"request-{len(calls)}.safetensors"
        if not calls:
            path.write_bytes(b"partial")
        else:
            save_file(_hidden_states(), path)
        calls.append(path)
        return str(path)

    monkeypatch.setattr(data_module, "generate_hidden_states", generate)
    dataset = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "index",
    )

    with pytest.warns(UserWarning, match="Failed to load/cache"):
        assert dataset._maybe_generate_hs(0) is None
    assert not calls[0].exists()
    assert list((shared_path / "artifacts").glob("*/*.safetensors")) == []

    loaded = dataset._maybe_generate_hs(0)

    assert loaded is not None
    assert len(calls) == 2
    assert not calls[1].exists()
    assert dataset.artifact_cache is not None
    stats = dataset.artifact_cache.snapshot_stats()
    assert stats["generation_failures"] == 1
    assert stats["publishes"] == 1


def test_shared_dataset_preserves_service_artifact_after_lock_timeout(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    service_path.mkdir()
    _write_dataset(data_path)
    artifact_path = service_path / "request.safetensors"
    artifact_path.write_bytes(b"partial")
    lock_path = artifact_path.parent / f"{artifact_path.name}.lock"
    lock_path.touch()

    def time_out_waiting_for_lock(*_args, **_kwargs):
        raise TimeoutError("active")

    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: str(artifact_path),
    )
    monkeypatch.setattr(
        transfer_module,
        "wait_for_lock",
        time_out_waiting_for_lock,
    )
    dataset = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "index",
    )

    with pytest.warns(UserWarning, match="Failed to load/cache"):
        assert dataset._maybe_generate_hs(0) is None

    assert artifact_path.exists()
    assert lock_path.exists()
    assert list((shared_path / "artifacts").glob("*/*.safetensors")) == []
    assert dataset.artifact_cache is not None
    stats = dataset.artifact_cache.snapshot_stats()
    assert stats["logical_requests"] == 1
    assert stats["misses"] == 1
    assert stats["generation_failures"] == 1
    assert stats["publishes"] == 0


def test_shared_dataset_preserves_on_generate_index_cache(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    index_path = tmp_path / "index"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        _successful_generator(service_path, calls),
    )
    dataset = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=index_path,
        on_generate="cache",
    )

    assert dataset._maybe_generate_hs(0) is not None

    indexed = index_path / "hs_0.safetensors"
    assert indexed.is_file()
    assert load_file(indexed)["token_ids"].tolist() == [1, 2, 3]
    assert len(calls) == 1


def test_train_and_validation_loaders_share_artifact_configuration(monkeypatch):
    dataset_kwargs = []

    def fake_arrow_dataset(**kwargs):
        dataset_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(dataloader_module, "ArrowDataset", fake_arrow_dataset)
    monkeypatch.setattr(dataloader_module, "get_dp_rank", lambda: 2)
    monkeypatch.setattr(
        dataloader_module,
        "_setup_dataloader",
        lambda dataset, *_args, **_kwargs: dataset,
    )

    dataloader_module.create_train_val_loaders(
        data_path="data",
        train_data_ratio=0.9,
        total_seq_len=128,
        hidden_states_dtype=torch.bfloat16,
        noise_std=0.0,
        legacy_data=False,
        transfer=None,
        vllm_endpoint="http://producer/v1",
        on_missing="generate",
        on_generate="delete",
        verifier_name_or_path="model",
        request_timeout=10,
        max_retries=2,
        hidden_size=4,
        num_target_layers=3,
        num_workers=0,
        prefetch_factor=1,
        preprocess=None,
        shared_artifacts_path="shared",
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_ttl_seconds=None,
        shared_artifacts_lock_timeout_seconds=45,
        shared_artifacts_consumer_id="consumer-a",
        shared_artifacts_lookbehind=3,
        shared_artifacts_lookahead=20,
        shared_artifacts_max_prefetch_per_consumer=7,
        shared_artifacts_capture_batch_size=6,
        shared_artifacts_capture_batch_wait_seconds=0.01,
        shared_artifacts_max_inflight=40,
        shared_artifacts_acquire_timeout_seconds=75,
        shared_artifacts_lease_timeout_seconds=600,
    )

    assert len(dataset_kwargs) == 2
    for kwargs in dataset_kwargs:
        assert kwargs["shared_artifacts_path"] == "shared"
        assert kwargs["shared_artifacts_namespace"] == "layers:2,18,33"
        assert kwargs["shared_artifacts_ttl_seconds"] is None
        assert kwargs["shared_artifacts_lock_timeout_seconds"] == 45
        assert kwargs["shared_artifacts_lookbehind"] == 3
        assert kwargs["shared_artifacts_lookahead"] == 20
        assert kwargs["shared_artifacts_max_prefetch_per_consumer"] == 7
        assert kwargs["shared_artifacts_capture_batch_size"] == 6
        assert kwargs["shared_artifacts_capture_batch_wait_seconds"] == 0.01
        assert kwargs["shared_artifacts_max_inflight"] == 40
        assert kwargs["shared_artifacts_acquire_timeout_seconds"] == 75
        assert kwargs["shared_artifacts_lease_timeout_seconds"] == 600
    assert dataset_kwargs[0]["shared_artifacts_consumer_id"] == "consumer-a:dp2:train"
    assert dataset_kwargs[1]["shared_artifacts_consumer_id"] == "consumer-a:dp2:val"


def test_train_only_loader_uses_full_dataset_without_validation(monkeypatch):
    dataset_kwargs = []
    dataset = object()

    def fake_arrow_dataset(**kwargs):
        dataset_kwargs.append(kwargs)
        return dataset

    monkeypatch.setattr(dataloader_module, "ArrowDataset", fake_arrow_dataset)
    monkeypatch.setattr(
        dataloader_module,
        "_setup_dataloader",
        lambda dataset, *_args, **_kwargs: dataset,
    )

    train_loader, val_loader = dataloader_module.create_train_val_loaders(
        data_path="data",
        train_data_ratio=1.0,
        total_seq_len=128,
        hidden_states_dtype=torch.bfloat16,
        noise_std=0.0,
        legacy_data=False,
        transfer=None,
        vllm_endpoint="http://producer/v1",
        on_missing="generate",
        on_generate="delete",
        verifier_name_or_path="model",
        request_timeout=10,
        max_retries=2,
        hidden_size=4,
        num_target_layers=3,
        num_workers=0,
        prefetch_factor=1,
        preprocess=None,
        shared_artifacts_path="shared",
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_consumer_id="consumer-a",
    )

    assert train_loader is dataset
    assert val_loader is None
    assert len(dataset_kwargs) == 1
    assert dataset_kwargs[0]["split_ratio"] == 1.0
    assert dataset_kwargs[0]["shared_artifacts_consumer_id"] == "consumer-a:dp0:train"
