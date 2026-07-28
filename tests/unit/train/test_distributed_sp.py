"""Multi-GPU tests for distributed sequence parallelism primitives.

These tests use ``torch.multiprocessing.spawn`` to create real process groups
and verify that the SP communication ops produce correct results.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.conftest import requires_multi_gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _dist_teardown():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# _init_sp_process_groups
# ---------------------------------------------------------------------------


def _worker_sp_group_init(rank, world_size, sp_size, results_dir):
    _dist_setup(rank, world_size)
    try:
        from speculators.train.distributed import (
            _init_sp_process_groups,
            get_dp_rank,
            get_dp_size,
            get_sp_group,
            get_sp_rank,
            get_sp_size,
        )

        _init_sp_process_groups(rank, world_size, sp_size)

        sp_group = get_sp_group()
        sp_ranks = dist.get_process_group_ranks(sp_group)

        torch.save(
            {
                "sp_size": get_sp_size(),
                "sp_rank": get_sp_rank(),
                "dp_size": get_dp_size(),
                "dp_rank": get_dp_rank(),
                "sp_group_ranks": sp_ranks,
            },
            results_dir / f"rank{rank}.pt",
        )
    finally:
        from speculators.train.distributed import maybe_destroy_distributed

        maybe_destroy_distributed()
        _dist_teardown()


@requires_multi_gpu
def test_sp_group_init_sp2(tmp_path):
    """With world_size=2, sp_size=2: one SP group {0,1}, one DP group per rank."""
    world_size = 2
    sp_size = 2
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_sp_group_init,
        args=(world_size, sp_size, results_dir),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = torch.load(results_dir / f"rank{rank}.pt", weights_only=False)
        assert r["sp_size"] == 2
        assert r["sp_rank"] == rank
        assert r["dp_size"] == 1
        assert r["dp_rank"] == 0
        assert r["sp_group_ranks"] == [0, 1]


@requires_multi_gpu
def test_sp_group_init_sp1(tmp_path):
    """With world_size=2, sp_size=1: no SP, each rank is its own SP group."""
    world_size = 2
    sp_size = 1
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_sp_group_init,
        args=(world_size, sp_size, results_dir),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = torch.load(results_dir / f"rank{rank}.pt", weights_only=False)
        assert r["sp_size"] == 1
        assert r["sp_rank"] == 0
        assert r["dp_size"] == 2
        assert r["dp_rank"] == rank
        assert r["sp_group_ranks"] == [rank]


# ---------------------------------------------------------------------------
# all_to_all_sp round-trip
# ---------------------------------------------------------------------------


def _worker_all_to_all_roundtrip(rank, world_size, results_dir):
    _dist_setup(rank, world_size)
    try:
        from speculators.train.distributed import _init_sp_process_groups

        sp_size = world_size
        _init_sp_process_groups(rank, world_size, sp_size)

        # Build a tensor where each rank owns a different chunk of heads
        # Input shape: (B=1, H=4, S_local=8, D=2)
        num_heads = sp_size * 2
        local_seq = 8
        head_dim = 2
        x = torch.randn(1, num_heads, local_seq, head_dim, device="cuda")
        x_orig = x.clone()

        from speculators.train.sequence_parallel import (
            ulysses_gather,
            ulysses_scatter,
        )
        from speculators.train.distributed import get_sp_group

        sp_group = get_sp_group()

        # scatter: (B, H, S_local, D) -> (B, H/sp, S_full, D)
        scattered = ulysses_scatter(x, sp_group, sp_size)
        assert scattered.shape == (1, num_heads // sp_size, local_seq * sp_size, head_dim)

        # gather: (B, H/sp, S_full, D) -> (B, H, S_local, D)
        gathered = ulysses_gather(scattered, sp_group, sp_size)
        assert gathered.shape == x_orig.shape

        torch.save(
            {"match": torch.allclose(gathered, x_orig, atol=1e-6)},
            results_dir / f"rank{rank}.pt",
        )
    finally:
        from speculators.train.distributed import maybe_destroy_distributed

        maybe_destroy_distributed()
        _dist_teardown()


@requires_multi_gpu
def test_all_to_all_roundtrip(tmp_path):
    """scatter then gather should recover the original tensor."""
    world_size = 2
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_all_to_all_roundtrip,
        args=(world_size, results_dir),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = torch.load(results_dir / f"rank{rank}.pt", weights_only=False)
        assert r["match"], f"Round-trip failed on rank {rank}"


# ---------------------------------------------------------------------------
# all_to_all_sp autograd backward
# ---------------------------------------------------------------------------


def _worker_all_to_all_backward(rank, world_size, results_dir):
    _dist_setup(rank, world_size)
    try:
        from speculators.train.distributed import _init_sp_process_groups

        sp_size = world_size
        _init_sp_process_groups(rank, world_size, sp_size)

        num_heads = sp_size * 2
        local_seq = 4
        head_dim = 2
        x = torch.randn(
            1, num_heads, local_seq, head_dim, device="cuda", requires_grad=True
        )

        # Forward: scatter then gather = identity
        from speculators.train.sequence_parallel import (
            ulysses_gather,
            ulysses_scatter,
        )
        from speculators.train.distributed import get_sp_group

        sp_group = get_sp_group()
        y = ulysses_gather(ulysses_scatter(x, sp_group, sp_size), sp_group, sp_size)
        loss = y.sum()
        loss.backward()

        # Gradient of sum w.r.t. input through an identity should be all-ones
        torch.save(
            {"grad_all_ones": torch.allclose(x.grad, torch.ones_like(x))},
            results_dir / f"rank{rank}.pt",
        )
    finally:
        from speculators.train.distributed import maybe_destroy_distributed

        maybe_destroy_distributed()
        _dist_teardown()


@requires_multi_gpu
def test_all_to_all_backward(tmp_path):
    """Backward through scatter+gather (identity) should produce all-ones gradient."""
    world_size = 2
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_all_to_all_backward,
        args=(world_size, results_dir),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = torch.load(results_dir / f"rank{rank}.pt", weights_only=False)
        assert r["grad_all_ones"], f"Backward failed on rank {rank}"


# ---------------------------------------------------------------------------
# register_sp_gradient_hooks
# ---------------------------------------------------------------------------


def _worker_sp_gradient_hooks(rank, world_size, results_dir):
    _dist_setup(rank, world_size)
    try:
        from speculators.train.distributed import (
            _init_sp_process_groups,
            register_sp_gradient_hooks,
        )

        sp_size = world_size
        _init_sp_process_groups(rank, world_size, sp_size)

        model = torch.nn.Linear(4, 4, bias=False).cuda()

        # Broadcast params so all ranks start identical
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        hooks = register_sp_gradient_hooks(model)
        assert len(hooks) > 0

        # Each rank computes a different "partial" loss
        x = torch.randn(1, 4, device="cuda") * (rank + 1)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # After the hook, each rank's .grad should be the SUM of all ranks' grads
        # Compute what the sum should be: run forward on each rank's data
        # and accumulate gradients manually
        grad = model.weight.grad.clone()

        # Clean up hooks
        for h in hooks:
            h.remove()

        torch.save({"grad": grad.cpu()}, results_dir / f"rank{rank}.pt")
    finally:
        from speculators.train.distributed import maybe_destroy_distributed

        maybe_destroy_distributed()
        _dist_teardown()


@requires_multi_gpu
def test_sp_gradient_hooks_sum_grads(tmp_path):
    """SP gradient hooks should all-reduce (sum) gradients across SP ranks."""
    world_size = 2
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_sp_gradient_hooks,
        args=(world_size, results_dir),
        nprocs=world_size,
        join=True,
    )

    r0 = torch.load(results_dir / "rank0.pt", weights_only=False)
    r1 = torch.load(results_dir / "rank1.pt", weights_only=False)

    # After all-reduce, both ranks should have identical gradients
    torch.testing.assert_close(r0["grad"], r1["grad"])
