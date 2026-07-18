from __future__ import annotations

import hashlib
import threading
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from speculators.data_generation.windowed_artifacts import (
    ArtifactGenerationError,
    ArtifactPriority,
    StreamSampleIndex,
    WindowedArtifactCoordinator,
    WindowedArtifactError,
    canonical_position_id,
    canonical_stream_id,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _samples(stream_id: str, count: int) -> tuple[StreamSampleIndex, ...]:
    return tuple(
        StreamSampleIndex(
            stream_id=stream_id,
            sequence=index,
            epoch=0,
            ordinal=index,
            dataset_index=index,
            batch_ordinal=index,
            batch_start_sequence=index,
            batch_end_sequence=index + 1,
            request_id=_digest(f"request-{index}"),
            position_id=canonical_position_id(
                stream_id,
                epoch=0,
                ordinal=index,
                dataset_index=index,
                batch_ordinal=index,
                batch_start_sequence=index,
                batch_end_sequence=index + 1,
            ),
        )
        for index in range(count)
    )


def _single_batch_samples(stream_id: str, count: int) -> tuple[StreamSampleIndex, ...]:
    return tuple(
        StreamSampleIndex(
            stream_id=stream_id,
            sequence=index,
            epoch=0,
            ordinal=index,
            dataset_index=index,
            batch_ordinal=0,
            batch_start_sequence=0,
            batch_end_sequence=count,
            request_id=_digest(f"request-{index}"),
            position_id=canonical_position_id(
                stream_id,
                epoch=0,
                ordinal=index,
                dataset_index=index,
                batch_ordinal=0,
                batch_start_sequence=0,
                batch_end_sequence=count,
            ),
        )
        for index in range(count)
    )


def _coordinator(tmp_path: Path, **kwargs) -> WindowedArtifactCoordinator:
    return WindowedArtifactCoordinator(
        tmp_path,
        poll_seconds=0.005,
        consumer_timeout_seconds=10,
        claim_timeout_seconds=10,
        **kwargs,
    )


def _register(
    coordinator: WindowedArtifactCoordinator,
    samples: tuple[StreamSampleIndex, ...],
    *consumer_ids: str,
    contract: dict | None = None,
    lookbehind: int = 0,
    lookahead: int = 2,
    max_prefetch: int | None = None,
    max_inflight: int = 4,
) -> None:
    contract = contract or {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    assert coordinator.register_stream(contract) == samples[0].stream_id
    coordinator.register_positions(samples)
    if max_prefetch is None:
        max_prefetch = min(8, lookahead + 1)
    for consumer_id in consumer_ids:
        coordinator.register_consumer(
            consumer_id,
            stream_id=samples[0].stream_id,
            lookbehind=lookbehind,
            lookahead=lookahead,
            max_prefetch=max_prefetch,
            max_inflight=max_inflight,
        )


def _publish(
    coordinator: WindowedArtifactCoordinator,
    stream_id: str,
    path: Path,
) -> str:
    claim = coordinator.claim_generation("producer", stream_id=stream_id)[0]
    artifact = path / f"{claim.request_id}.safetensors"
    artifact.write_bytes(b"payload")
    coordinator.complete_generation(
        "producer", claim, path=artifact, size_bytes=artifact.stat().st_size
    )
    return claim.request_id


def _complete_claim(
    coordinator: WindowedArtifactCoordinator,
    owner: str,
    claim,
    path: Path,
) -> None:
    artifact = path / f"{claim.request_id}.safetensors"
    artifact.write_bytes(b"payload")
    coordinator.complete_generation(
        owner, claim, path=artifact, size_bytes=artifact.stat().st_size
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        time.sleep(0.005)


def test_stream_and_position_identity_cover_order_contract():
    contract = {
        "dataset_fingerprint": "dataset-a",
        "epoch_order": "multipack-v2",
        "sampler_seed": 7,
    }
    stream_id = canonical_stream_id(contract)

    assert stream_id == canonical_stream_id(dict(reversed(tuple(contract.items()))))
    assert stream_id != canonical_stream_id({**contract, "sampler_seed": 8})
    assert canonical_position_id(
        stream_id,
        epoch=2,
        ordinal=3,
        dataset_index=9,
        batch_ordinal=1,
        batch_start_sequence=3,
        batch_end_sequence=5,
    ) != canonical_position_id(
        stream_id,
        epoch=2,
        ordinal=4,
        dataset_index=9,
        batch_ordinal=1,
        batch_start_sequence=3,
        batch_end_sequence=5,
    )


def test_existing_schema_connection_does_not_wait_for_writer(tmp_path):
    owner = _coordinator(tmp_path)
    owner._conn.execute("BEGIN IMMEDIATE")
    finished = threading.Event()
    errors: list[BaseException] = []

    def connect() -> None:
        try:
            with _coordinator(tmp_path):
                pass
        except BaseException as error:  # noqa: BLE001 - thread boundary
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=connect)
    thread.start()
    try:
        assert finished.wait(0.5), "existing schema initialization attempted a write"
    finally:
        owner._conn.rollback()
        owner.close()
        thread.join(timeout=2)
    assert not errors


def test_completed_consumer_reactivates_at_its_committed_cursor(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 1)
    coordinator = _coordinator(tmp_path)
    _register(coordinator, samples, "consumer")

    coordinator.complete_consumer("consumer")
    coordinator.register_consumer(
        "consumer",
        stream_id=stream_id,
        lookbehind=0,
        lookahead=2,
        max_prefetch=3,
        max_inflight=4,
        cursor=0,
    )

    assert coordinator.snapshot()["consumers"][0]["state"] == "active"


def test_max_prefetch_is_part_of_the_persisted_resume_contract(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 4)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "consumer",
        lookahead=3,
        max_prefetch=2,
    )
    coordinator.close()

    with _coordinator(tmp_path) as resumed:
        resumed.register_consumer(
            "consumer",
            stream_id=stream_id,
            lookbehind=0,
            lookahead=3,
            max_prefetch=2,
            max_inflight=4,
        )
        assert resumed.snapshot()["consumers"][0]["max_prefetch"] == 2
        with pytest.raises(WindowedArtifactError, match="configuration changed"):
            resumed.register_consumer(
                "consumer",
                stream_id=stream_id,
                lookbehind=0,
                lookahead=3,
                max_prefetch=3,
                max_inflight=4,
            )


def test_two_consumers_share_publication_but_commit_independently(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 3)
    coordinator = _coordinator(tmp_path)
    _register(coordinator, samples, "consumer-a", "consumer-b")
    _publish(coordinator, stream_id, tmp_path)

    first = coordinator.acquire("consumer-a", samples[0], timeout_seconds=1)
    second = coordinator.acquire("consumer-b", samples[0], timeout_seconds=1)
    assert not first.cache_hit
    assert second.cache_hit

    assert coordinator.ack("consumer-a", [first.as_batch_metadata()]) == 1
    snapshot = coordinator.snapshot()
    cursors = {row["consumer_id"]: row["cursor"] for row in snapshot["consumers"]}
    assert cursors == {"consumer-a": 1, "consumer-b": 0}
    assert snapshot["high_water"] == {
        "inflight_acquisitions": 2,
        "retained_artifacts": 1,
        "retained_bytes": len(b"payload"),
    }
    assert coordinator.begin_evictions() == ()

    assert coordinator.ack("consumer-b", [second.as_batch_metadata()]) == 1
    evictions = coordinator.begin_evictions()
    assert [claim.request_id for claim in evictions] == [samples[0].request_id]


def test_prefetch_cap_limits_active_work_and_tops_up_after_completion(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 50)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "consumer",
        lookahead=40,
        max_prefetch=8,
    )

    snapshot = coordinator.snapshot()
    assert snapshot["artifact_states"] == {"absent": 42, "queued": 8}
    first = coordinator.claim_generation(
        "producer-a",
        stream_id=stream_id,
        max_claims=8,
        max_active_claims=8,
    )
    assert len(first) == 8
    assert all(claim.priority is ArtifactPriority.PREFETCH for claim in first)
    assert (
        coordinator.claim_generation(
            "producer-b",
            stream_id=stream_id,
            max_claims=8,
            max_active_claims=8,
        )
        == ()
    )

    _complete_claim(coordinator, "producer-a", first[0], tmp_path)
    assert coordinator.snapshot()["artifact_states"] == {
        "absent": 41,
        "generating": 7,
        "queued": 1,
        "ready": 1,
    }
    second = coordinator.claim_generation(
        "producer-b",
        stream_id=stream_id,
        max_claims=8,
        max_active_claims=8,
    )
    assert len(second) == 1
    assert coordinator.snapshot()["artifact_states"]["generating"] == 8


def test_demand_bypasses_full_prefetch_cap(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 50)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "consumer",
        lookahead=40,
        max_prefetch=8,
    )
    acquired: list = []
    thread = threading.Thread(
        target=lambda: acquired.append(
            coordinator.acquire("consumer", samples[20], timeout_seconds=2)
        )
    )
    thread.start()
    _wait_until(lambda: coordinator.snapshot()["inflight_acquisitions"] == 1)

    claims = coordinator.claim_generation("producer", stream_id=stream_id, max_claims=9)
    assert len(claims) == 9
    assert claims[0].request_id == samples[20].request_id
    assert claims[0].priority is ArtifactPriority.DEMAND
    for claim in claims:
        _complete_claim(coordinator, "producer", claim, tmp_path)
    thread.join(2)
    assert len(acquired) == 1
    coordinator.ack("consumer", [acquired[0].as_batch_metadata()])


def test_max_inflight_is_independent_of_dataloader_prefetch(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 3)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "consumer",
        lookahead=2,
        max_inflight=1,
    )
    while coordinator.snapshot()["artifact_states"].get("queued", 0):
        _publish(coordinator, stream_id, tmp_path)

    first = coordinator.acquire("consumer", samples[0], timeout_seconds=1)
    acquired: list = []
    thread = threading.Thread(
        target=lambda: acquired.append(
            coordinator.acquire("consumer", samples[1], timeout_seconds=2)
        )
    )
    thread.start()
    time.sleep(0.05)
    assert acquired == []

    coordinator.ack("consumer", [first.as_batch_metadata()])
    thread.join(2)
    assert len(acquired) == 1
    coordinator.ack("consumer", [acquired[0].as_batch_metadata()])


def test_authorized_batch_finishes_when_larger_than_window_limits(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _single_batch_samples(stream_id, 3)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "consumer",
        lookahead=0,
        max_inflight=1,
    )
    while coordinator.snapshot()["artifact_states"].get("queued", 0):
        _publish(coordinator, stream_id, tmp_path)

    def acquire_one(current, results) -> None:
        results.append(coordinator.acquire("consumer", current, timeout_seconds=1))

    leases = [coordinator.acquire("consumer", samples[0], timeout_seconds=1)]
    for sample in samples[1:]:
        acquired: list = []
        thread = threading.Thread(
            target=acquire_one,
            args=(sample, acquired),
        )
        thread.start()
        _wait_until(
            lambda: coordinator.snapshot()["inflight_acquisitions"] == len(leases) + 1
        )
        _publish(coordinator, stream_id, tmp_path)
        thread.join(2)
        assert len(acquired) == 1
        leases.extend(acquired)
    assert coordinator.snapshot()["inflight_acquisitions"] == 3
    assert (
        coordinator.ack("consumer", [lease.as_batch_metadata() for lease in leases])
        == 3
    )


def test_demand_claims_are_round_robin_across_consumers(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 6)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "a",
        "b",
        "c",
        lookahead=5,
        max_inflight=2,
    )
    errors: list[Exception] = []

    def acquire(consumer: str, sample: StreamSampleIndex) -> None:
        try:
            coordinator.acquire(consumer, sample, timeout_seconds=1)
        except (TimeoutError, ArtifactGenerationError):
            pass
        except Exception as error:  # noqa: BLE001 - surface worker failures in test
            errors.append(error)

    threads = [
        threading.Thread(target=acquire, args=(consumer, samples[index]))
        for index, consumer in enumerate(("a", "b", "c"))
    ]
    for thread in threads:
        thread.start()
    _wait_until(lambda: coordinator.snapshot()["inflight_acquisitions"] == 3)

    claims = coordinator.claim_generation("producer", stream_id=stream_id, max_claims=3)
    assert {claim.request_id for claim in claims} == {
        samples[0].request_id,
        samples[1].request_id,
        samples[2].request_id,
    }
    assert all(claim.priority is ArtifactPriority.DEMAND for claim in claims)
    for claim in claims:
        coordinator.fail_generation("producer", claim, "stop test")
    for thread in threads:
        thread.join(2)
    assert errors == []


def test_generation_failure_retries_then_wakes_waiter(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 1)
    coordinator = _coordinator(tmp_path, max_generation_attempts=2)
    _register(coordinator, samples, "consumer", lookahead=0)
    result: list[Exception] = []

    def wait() -> None:
        try:
            coordinator.acquire("consumer", samples[0], timeout_seconds=2)
        except Exception as error:  # noqa: BLE001 - assert exact async failure below
            result.append(error)

    thread = threading.Thread(target=wait)
    thread.start()
    _wait_until(lambda: coordinator.snapshot()["inflight_acquisitions"] == 1)
    first = coordinator.claim_generation("producer", stream_id=stream_id)[0]
    coordinator.fail_generation("producer", first, "first failure")
    second = coordinator.claim_generation("producer", stream_id=stream_id)[0]
    coordinator.fail_generation("producer", second, "terminal failure")
    thread.join(2)

    assert len(result) == 1
    assert isinstance(result[0], ArtifactGenerationError)
    assert "terminal failure" in str(result[0])


def test_terminal_prefetch_failure_releases_capacity_for_next_position(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 5)
    coordinator = _coordinator(tmp_path, max_generation_attempts=1)
    _register(
        coordinator,
        samples,
        "consumer",
        lookahead=4,
        max_prefetch=2,
    )

    claims = coordinator.claim_generation("producer", stream_id=stream_id, max_claims=2)
    coordinator.fail_generation("producer", claims[0], "terminal failure")

    assert coordinator.snapshot()["artifact_states"] == {
        "absent": 2,
        "failed": 1,
        "generating": 1,
        "queued": 1,
    }


def test_expired_consumer_releases_window_and_read_lease(tmp_path):
    now = [100.0]
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 1)
    coordinator = WindowedArtifactCoordinator(
        tmp_path,
        poll_seconds=0.005,
        consumer_timeout_seconds=5,
        claim_timeout_seconds=10,
        clock=lambda: now[0],
    )
    _register(coordinator, samples, "consumer", lookahead=0)
    _publish(coordinator, stream_id, tmp_path)
    coordinator.acquire("consumer", samples[0], timeout_seconds=1)

    now[0] += 6
    assert coordinator.recover_expired() == {
        "expired_consumers": 1,
        "expired_claims": 0,
    }
    assert coordinator.snapshot()["inflight_acquisitions"] == 0
    assert len(coordinator.begin_evictions()) == 1


def test_resume_reset_rewinds_cursor_and_clears_uncommitted_leases(tmp_path):
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 2)
    coordinator = _coordinator(tmp_path)
    _register(coordinator, samples, "consumer", lookahead=1, max_inflight=2)
    _publish(coordinator, stream_id, tmp_path)
    _publish(coordinator, stream_id, tmp_path)
    committed = coordinator.acquire("consumer", samples[0], timeout_seconds=1)
    assert coordinator.ack("consumer", [committed.as_batch_metadata()]) == 1
    uncommitted = coordinator.acquire("consumer", samples[1], timeout_seconds=1)
    assert coordinator.snapshot()["inflight_acquisitions"] == 1

    coordinator.register_consumer(
        "consumer",
        stream_id=stream_id,
        lookbehind=0,
        lookahead=1,
        max_prefetch=2,
        max_inflight=2,
        cursor=0,
        reset=True,
    )
    snapshot = coordinator.snapshot()
    assert snapshot["consumers"][0]["cursor"] == 0
    assert snapshot["inflight_acquisitions"] == 0
    with pytest.raises(WindowedArtifactError, match="unknown, stale, or mismatched"):
        coordinator.ack("consumer", [uncommitted.as_batch_metadata()])
    replay = coordinator.acquire("consumer", samples[0], timeout_seconds=1)
    assert coordinator.ack("consumer", [replay.as_batch_metadata()]) == 1


def test_expired_producer_claim_is_reassigned_with_bounded_attempts(tmp_path):
    now = [100.0]
    contract = {"dataset_fingerprint": "dataset-a", "sampler_seed": 0}
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, 1)
    coordinator = WindowedArtifactCoordinator(
        tmp_path,
        poll_seconds=0.005,
        consumer_timeout_seconds=30,
        claim_timeout_seconds=5,
        max_generation_attempts=2,
        clock=lambda: now[0],
    )
    _register(coordinator, samples, "consumer", lookahead=0)
    first = coordinator.claim_generation("producer-a", stream_id=stream_id)[0]
    assert first.generation == 0

    now[0] += 6
    assert coordinator.recover_expired()["expired_claims"] == 1
    second = coordinator.claim_generation("producer-b", stream_id=stream_id)[0]
    assert second.generation == 1
    now[0] += 6
    coordinator.recover_expired()

    assert coordinator.claim_generation("producer-c", stream_id=stream_id) == ()
    assert coordinator.snapshot()["artifact_states"] == {"failed": 1}


def _assert_long_stream_retention_bound(tmp_path: Path, count: int) -> None:
    contract = {
        "dataset_fingerprint": f"dataset-{count}",
        "sampler_seed": 0,
    }
    stream_id = canonical_stream_id(contract)
    samples = _samples(stream_id, count)
    coordinator = _coordinator(tmp_path)
    _register(
        coordinator,
        samples,
        "consumer",
        contract=contract,
        lookbehind=2,
        lookahead=16,
        max_inflight=32,
    )
    retention_bound = 2 + 16 + 1

    for cursor in (0, count // 2, count - 1):
        coordinator.register_consumer(
            "consumer",
            stream_id=stream_id,
            lookbehind=2,
            lookahead=16,
            max_prefetch=8,
            max_inflight=32,
            cursor=cursor,
            reset=True,
        )
        for eviction in coordinator.begin_evictions(limit=retention_bound * 2):
            coordinator.finish_eviction(eviction, removed=True)
        while coordinator.snapshot()["artifact_states"].get("queued", 0):
            _publish(coordinator, stream_id, tmp_path)
        snapshot = coordinator.snapshot()
        assert snapshot["retained_artifacts"] <= retention_bound
        assert snapshot["artifact_states"].get("queued", 0) == 0
        for eviction in coordinator.begin_evictions(limit=retention_bound * 2):
            coordinator.finish_eviction(eviction, removed=True)


def test_10k_position_stream_has_window_bounded_payload_retention(tmp_path):
    _assert_long_stream_retention_bound(tmp_path, 10_000)


@pytest.mark.slow
def test_100k_position_stream_has_window_bounded_payload_retention(tmp_path):
    _assert_long_stream_retention_bound(tmp_path, 100_000)
