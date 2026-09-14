import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import speculators.train.recovery as recovery_module
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    BatchRecoveryCoordinator,
    GenerationRecoveryGuard,
    RecoveryMetadata,
    SampleUnavailable,
)


def _ddp_local_empty_worker(rank, world_size, init_file, output_dir):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(torch.nn.Linear(2, 1, bias=False))
        output = model(torch.ones(1, 2)).sum()
        # Rank 0 models an all-failed local batch; rank 1 retains useful data.
        loss = output * (0.0 if rank == 0 else 1.0)
        loss.backward()
        torch.save(model.module.weight.grad, f"{output_dir}/grad-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_generation_guard_counts_exhausted_samples_not_attempts():
    guard = GenerationRecoveryGuard(retries=2, max_consecutive_failures=2)
    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise ValueError("bad payload")

    first = guard.run(fail, description="round trip failed")
    second = guard.run(fail, description="round trip failed")

    assert isinstance(first, SampleUnavailable)
    assert first.consecutive_failures == 1
    assert not first.fatal
    assert isinstance(second, SampleUnavailable)
    assert second.consecutive_failures == 2
    assert second.fatal
    assert attempts == 6


def test_generation_guard_resets_after_a_valid_sample():
    guard = GenerationRecoveryGuard(retries=0, max_consecutive_failures=2)

    def fail():
        raise ValueError("bad payload")

    unavailable = guard.run(fail, description="round trip failed")
    result = guard.run(lambda: "valid", description="round trip failed")

    assert isinstance(unavailable, SampleUnavailable)
    assert result == "valid"
    assert guard.consecutive_failures == 0


def test_locally_empty_batch_does_not_mask_other_ranks_or_skip_step():
    batch: dict[str, Any] = {
        "loss_mask": torch.zeros(1, 8),
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        RECOVERY_METADATA_KEY: RecoveryMetadata(
            failure_count=1,
            locally_empty=True,
        ),
    }

    BatchRecoveryCoordinator("training").consume(batch)

    assert batch["loss_mask"].shape == (1, 8)
    assert not batch["loss_mask"].bool().any()
    assert batch["input_ids"].shape == (1, 8)
    assert not any(key.startswith("__") for key in batch)


def test_remote_circuit_breaker_stops_this_rank_at_status_collective(monkeypatch):
    batch = {
        "loss_mask": torch.ones(1, 8),
        RECOVERY_METADATA_KEY: RecoveryMetadata(),
    }

    def emulate_remote_fatal(status, op):
        assert op == recovery_module.dist.ReduceOp.SUM
        status[0] += 1

    monkeypatch.setattr(recovery_module, "is_distributed", lambda: True)
    monkeypatch.setattr(recovery_module.dist, "all_reduce", emulate_remote_fatal)

    with pytest.raises(RuntimeError, match="circuit breaker tripped on 1 rank"):
        BatchRecoveryCoordinator("training", device=torch.device("cpu")).consume(
            batch,
            synchronize=True,
        )


def test_recovery_status_defaults_to_cpu_without_accelerator(monkeypatch):
    batch = {RECOVERY_METADATA_KEY: RecoveryMetadata(failure_count=1)}

    def check_cpu_status(status, op):
        assert status.device.type == "cpu"
        assert op == recovery_module.dist.ReduceOp.SUM

    monkeypatch.setattr(recovery_module, "is_distributed", lambda: True)
    monkeypatch.setattr(
        recovery_module.torch.accelerator,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(recovery_module.dist, "all_reduce", check_cpu_status)

    BatchRecoveryCoordinator("training").consume(batch, synchronize=True)


def test_recovery_metadata_accumulates_until_synchronization(monkeypatch):
    statuses = []

    def record_status(status, op):
        assert op == recovery_module.dist.ReduceOp.SUM
        statuses.append(status.tolist())

    monkeypatch.setattr(recovery_module, "is_distributed", lambda: True)
    monkeypatch.setattr(recovery_module.dist, "all_reduce", record_status)
    coordinator = BatchRecoveryCoordinator("training", device=torch.device("cpu"))

    coordinator.consume({RECOVERY_METADATA_KEY: RecoveryMetadata(failure_count=1)})
    assert not statuses

    coordinator.consume(
        {RECOVERY_METADATA_KEY: RecoveryMetadata(failure_count=2)},
        synchronize=True,
    )
    assert statuses == [[0, 3]]


def test_unknown_metadata_key_fails_before_reaching_the_model():
    batch = {"loss_mask": torch.ones(1, 8), "__generation_future_key__": True}

    with pytest.raises(RuntimeError, match="__generation_future_key__"):
        BatchRecoveryCoordinator("training").consume(batch)


@pytest.mark.slow
def test_ddp_local_zero_loss_still_joins_backward_collectives(tmp_path):
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _ddp_local_empty_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    for rank in range(2):
        grad = torch.load(tmp_path / f"grad-{rank}.pt", weights_only=True)
        # DDP averages rank 0's local zero gradient with rank 1's unit gradient.
        assert torch.equal(grad, torch.full((1, 2), 0.5))
