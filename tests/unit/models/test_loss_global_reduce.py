"""Tests for global (cross-rank) loss normalization in ``loss_function``.

Two invariants: non-distributed behavior is unchanged, and under UNEVEN per-rank
token counts the DDP-averaged gradient equals the analytic global token-weighted
gradient ``∂(Σ_r Σ_t loss)/(Σ_r Σ_t mask)``.
"""

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from speculators.models.metrics import loss_function, tv_loss

_V = 8
_EPS = 1e-5


def _rank_data(rank: int):
    """Deterministic per-rank batch; token count varies with rank ⇒ uneven denoms."""
    gen = torch.Generator().manual_seed(100 + rank)
    seq_len = 4 + 2 * rank
    logits = torch.randn(1, seq_len, _V, generator=gen)
    targets = torch.randn(1, seq_len, _V, generator=gen)
    loss_mask = torch.ones(1, seq_len)
    pos_idx = torch.zeros(1, seq_len)
    return logits, targets, loss_mask, pos_idx


def _analytic_global_grad(world_size: int) -> torch.Tensor:
    """Grad of the true global objective Σ_r Σ_t loss / Σ_r Σ_t mask, on one process."""
    theta = torch.nn.Parameter(torch.ones(_V))
    num = torch.zeros(())
    den = torch.zeros(())
    for rank in range(world_size):
        logits, targets, loss_mask, _ = _rank_data(rank)
        elementwise = tv_loss(logits * theta, targets) * loss_mask
        num = num + elementwise.sum()
        den = den + loss_mask.sum()
    (num / (den + _EPS)).backward()
    return theta.grad.detach().clone()


def _worker(rank: int, world_size: int, init_file: str, ret: dict) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    theta = torch.nn.Parameter(torch.ones(_V))  # identical shared param on every rank
    logits, targets, loss_mask, pos_idx = _rank_data(rank)
    loss_function(
        logits * theta, targets, loss_mask, pos_idx, loss_fn=tv_loss
    ).backward()

    grad = theta.grad.detach().clone()
    dist.all_reduce(
        grad, op=dist.ReduceOp.SUM
    )  # DDP/FSDP mean-average of per-rank grads
    grad /= world_size

    if rank == 0:
        analytic = _analytic_global_grad(world_size)
        ret["max_abs_err"] = (grad - analytic).abs().max().item()
    dist.destroy_process_group()


def test_single_process_is_unchanged():
    """No process group ⇒ the rank-local masked-mean path (unchanged upstream)."""
    logits, targets, loss_mask, pos_idx = _rank_data(1)
    got = loss_function(logits, targets, loss_mask, pos_idx, loss_fn=tv_loss)

    elementwise = tv_loss(logits, targets) * loss_mask
    expected = (elementwise.sum(dim=1) / (loss_mask.sum(dim=1) + _EPS)).mean()
    assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)
def test_ddp_averaged_grad_equals_global_objective_under_imbalance():
    """2 ranks, UNEVEN token counts: mean-averaged grad == the global objective's."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        init_file = f"{tmp}/pg_init"
        manager = mp.Manager()
        ret = manager.dict()
        mp.spawn(
            _worker, args=(world_size, init_file, ret), nprocs=world_size, join=True
        )
    assert "max_abs_err" in ret, "rank-0 worker did not report a result"
    assert ret["max_abs_err"] < 1e-5, (
        f"grad differs from the global objective: {ret['max_abs_err']}"
    )
