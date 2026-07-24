"""Unit tests for gradient accumulation.

Covers the two correctness-critical pieces of the feature without spinning up a
full ``Trainer.train_epoch`` run:

- The numerical identity that N accumulated ``(loss / N).backward()`` micro-steps
  produce the same gradient as one backward over the concatenated batch (this is
  why loss is scaled by ``1 / gradient_accumulation_steps`` in the loop).
- ``Trainer._maybe_no_sync`` returns a real ``no_sync`` context only for a DDP model
  on a non-boundary micro-step, and a no-op context otherwise.
"""

import contextlib
import os
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from speculators.model import SpeculatorModel
from speculators.train.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Numerical equivalence: accumulate == single large batch
# ---------------------------------------------------------------------------


def _grad_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
    )


@pytest.mark.parametrize(
    ("accum", "batch"),
    [(1, 4), (4, 4), (8, 3)],
)
def test_accumulated_grad_matches_single_large_batch(accum: int, batch: int) -> None:
    """N micro-steps of ``(loss / N).backward()`` == one backward over the whole
    batch with mean-reduction loss (grads accumulate additively into ``.grad``)."""
    torch.manual_seed(0)
    in_dim, out_dim = 5, 3
    model = nn.Linear(in_dim, out_dim).double()
    loss_fn = nn.MSELoss()  # mean reduction

    # Equal-size micro-batches so mean(concat) == (1/N) * sum(mean(chunk_i)).
    xs = [torch.randn(batch, in_dim, dtype=torch.float64) for _ in range(accum)]
    ys = [torch.randn(batch, out_dim, dtype=torch.float64) for _ in range(accum)]

    # Accumulation path (mirrors train_epoch: zero at window start, loss / accum).
    model.zero_grad(set_to_none=True)
    for x, y in zip(xs, ys, strict=True):
        (loss_fn(model(x), y) / accum).backward()
    acc_grad = _grad_vector(model)

    # Reference: single backward over the concatenated batch.
    model.zero_grad(set_to_none=True)
    loss_fn(model(torch.cat(xs)), torch.cat(ys)).backward()
    ref_grad = _grad_vector(model)

    assert torch.allclose(acc_grad, ref_grad, atol=1e-10, rtol=1e-8)


def test_accum_one_is_exact_single_step() -> None:
    """accum=1 must be bit-for-bit identical to a plain single backward."""
    torch.manual_seed(1)
    model = nn.Linear(4, 2).double()
    x = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randn(6, 2, dtype=torch.float64)
    loss_fn = nn.MSELoss()

    model.zero_grad(set_to_none=True)
    (loss_fn(model(x), y) / 1).backward()
    acc_grad = _grad_vector(model)

    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    plain_grad = _grad_vector(model)

    assert torch.equal(acc_grad, plain_grad)


# ---------------------------------------------------------------------------
# _maybe_no_sync
# ---------------------------------------------------------------------------


def _call_maybe_no_sync(model: nn.Module, is_boundary: bool):
    # Bind the unbound method to a minimal stand-in carrying only `.model`.
    stand_in = cast("Trainer", SimpleNamespace(model=model))
    return Trainer._maybe_no_sync(stand_in, is_boundary)


def test_maybe_no_sync_plain_module_is_nullcontext() -> None:
    model = nn.Linear(2, 2)
    for is_boundary in (True, False):
        ctx = _call_maybe_no_sync(model, is_boundary)
        assert isinstance(ctx, contextlib.nullcontext)


@pytest.fixture
def single_process_group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_maybe_no_sync_ddp_skips_sync_off_boundary(single_process_group) -> None:
    ddp = DistributedDataParallel(nn.Linear(2, 2))

    # Non-boundary micro-step: a real no_sync context (not the no-op).
    off_boundary = _call_maybe_no_sync(ddp, is_boundary=False)
    assert not isinstance(off_boundary, contextlib.nullcontext)

    # Boundary micro-step: sync must happen, so a no-op context.
    on_boundary = _call_maybe_no_sync(ddp, is_boundary=True)
    assert isinstance(on_boundary, contextlib.nullcontext)


# ---------------------------------------------------------------------------
# Validation: accum larger than an epoch is a hard error (never steps)
# ---------------------------------------------------------------------------


def test_trainer_rejects_accum_larger_than_epoch() -> None:
    """accum > batches-per-epoch would run zero optimizer steps; reject it early."""
    loader = DataLoader(TensorDataset(torch.arange(3)), batch_size=1)  # 3 batches
    cfg = TrainerConfig(
        lr=1e-4,
        num_epochs=1,
        save_path="unused",  # error raises before any checkpoint I/O
        gradient_accumulation_steps=5,
    )
    with pytest.raises(ValueError, match="exceeds the number of"):
        Trainer(cast("SpeculatorModel", nn.Identity()), cfg, loader)


@pytest.mark.parametrize("accum", [0, -1])
def test_trainer_rejects_non_positive_accum(accum: int) -> None:
    """The Trainer API guards accum >= 1 even when constructed directly (bypassing
    the CLI validator): 0 would ZeroDivisionError and negatives negate gradients."""
    loader = DataLoader(TensorDataset(torch.arange(3)), batch_size=1)
    cfg = TrainerConfig(
        lr=1e-4,
        num_epochs=1,
        save_path="unused",
        gradient_accumulation_steps=accum,
    )
    with pytest.raises(ValueError, match="must be >= 1"):
        Trainer(cast("SpeculatorModel", nn.Identity()), cfg, loader)


# ---------------------------------------------------------------------------
# Sub-epoch checkpoint cadence is measured in optimizer steps
# ---------------------------------------------------------------------------


def test_step_checkpoint_cadence_uses_optimizer_steps() -> None:
    """Mid-epoch checkpoints fire on the optimizer-step cadence and stay aligned to
    accumulation boundaries. With 100 microbatches, accum=4, checkpoint_freq=0.5 the
    interval is 12 optimizer steps, so the only qualifying save is at optimizer step
    12 (microbatch 48); step 24 is dropped by the end-of-epoch guard."""
    num_steps, accum = 100, 4
    opt_steps_per_epoch = num_steps // accum  # 25
    step_interval = max(1, round(opt_steps_per_epoch * 0.5))  # 12

    saved: list[int] = []
    stand_in = cast(
        "Trainer",
        SimpleNamespace(
            config=SimpleNamespace(save_best=False),
            maybe_save_checkpoint=lambda epoch, local_step: saved.append(local_step),
        ),
    )
    for local_step in range(1, num_steps + 1):
        Trainer._maybe_save_step_checkpoint(
            stand_in,
            0,
            local_step,
            accum,
            opt_steps_per_epoch,
            step_interval,
            local_step % accum == 0,
        )

    assert saved == [48]
