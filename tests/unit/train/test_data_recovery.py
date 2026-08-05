import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import speculators.train.utils as utils_module
from speculators.train.utils import consume_recovery_metadata


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


def test_locally_empty_batch_does_not_mask_other_ranks_or_skip_step():
    batch = {
        "loss_mask": torch.zeros(1, 8),
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "__locally_empty_batch__": True,
        "__generation_failure_count__": 1,
    }

    consume_recovery_metadata(batch, phase="training")

    assert batch["loss_mask"].shape == (1, 8)
    assert not batch["loss_mask"].bool().any()
    assert batch["input_ids"].shape == (1, 8)
    assert not any(key.startswith("__") for key in batch)


def test_remote_circuit_breaker_stops_this_rank_at_status_collective(monkeypatch):
    batch = {
        "loss_mask": torch.ones(1, 8),
        "__generation_failure_count__": 0,
    }

    def emulate_remote_fatal(status, op):
        assert op == utils_module.dist.ReduceOp.SUM
        status[0] += 1

    monkeypatch.setattr(utils_module, "is_distributed", lambda: True)
    monkeypatch.setattr(utils_module.dist, "all_reduce", emulate_remote_fatal)

    with pytest.raises(RuntimeError, match="circuit breaker tripped on 1 rank"):
        # Explicit CPU device: the real default is this rank's accelerator.
        consume_recovery_metadata(batch, phase="training", device="cpu")


def test_unknown_metadata_key_fails_before_reaching_the_model():
    batch = {"loss_mask": torch.ones(1, 8), "__generation_future_key__": True}

    with pytest.raises(RuntimeError, match="__generation_future_key__"):
        consume_recovery_metadata(batch, phase="training")


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
