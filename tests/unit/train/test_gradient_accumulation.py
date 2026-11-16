"""Tests for gradient accumulation in the trainer."""

import os
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

import speculators.train.trainer as trainer_module
from speculators.model import SpeculatorModel
from speculators.train.trainer import Trainer, TrainerConfig


@pytest.fixture
def single_process_group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_maybe_no_sync_fsdp2_toggles_sync_at_boundary(monkeypatch) -> None:
    class FakeFSDPModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.sync_values: list[bool] = []

        def set_requires_gradient_sync(
            self, requires_gradient_sync: bool, *, recurse: bool = True
        ) -> None:
            assert recurse
            self.sync_values.append(requires_gradient_sync)

    monkeypatch.setattr(trainer_module, "FSDPModule", FakeFSDPModule)
    model = FakeFSDPModule()

    trainer = cast("Trainer", SimpleNamespace(model=model))
    Trainer._maybe_no_sync(trainer, False)
    Trainer._maybe_no_sync(trainer, True)
    assert model.sync_values == [False, True]


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
            config=SimpleNamespace(save_best=False, gradient_accumulation_steps=accum),
            maybe_save_checkpoint=lambda epoch, local_step: saved.append(local_step),
        ),
    )
    for local_step in range(accum, num_steps + 1, accum):
        Trainer._maybe_save_step_checkpoint(
            stand_in,
            0,
            local_step,
            opt_steps_per_epoch,
            step_interval,
        )

    assert saved == [48]


@pytest.mark.parametrize(
    ("accum", "max_steps", "updates"),
    [(1, None, 9), (4, None, 2), (4, 1, 1)],
)
def test_train_epoch_matches_large_batch_update(
    single_process_group, monkeypatch, accum, max_steps, updates
):
    """Check weight updates, communication and metrics through the training loop."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.25))
            self.forwards = 0

        def forward(self, x, error_records, document_ids):
            self.forwards += 1
            return (
                None,
                (self.weight * (x / 10)).square().mean(),
                {
                    "accuracy_sum": x.sum(),
                    "accuracy_total": x.square().sum(),
                    "plain_metric": x.mean(),
                },
            )

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    logged = []
    monkeypatch.setattr(
        trainer_module.metric_logger, "info", lambda data, **kw: logged.append(data)
    )
    monkeypatch.setattr(trainer_module, "tqdm", lambda loader, **kw: loader)
    model = Model()
    ddp = DistributedDataParallel(model)
    syncs = []

    def count_sync(state, bucket):
        syncs.append(1)
        future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        future.set_result(bucket.buffer())
        return future

    ddp.register_comm_hook(None, count_sync)
    trainer = Trainer.__new__(Trainer)
    trainer.model = cast("SpeculatorModel", ddp)
    trainer.config = TrainerConfig(
        lr=0.01,
        num_epochs=1,
        save_path="unused",
        gradient_accumulation_steps=accum,
        log_freq=1,
        max_steps=max_steps,
        hidden_states_dtype=torch.bfloat16,
    )

    class Loader:
        batch_sampler = None

        def __len__(self):
            return 9

        def __iter__(self):
            for i in range(1, 10):
                yield {
                    "x": torch.full((2,), float(i)),
                    "document_ids": torch.zeros(2, dtype=torch.long),
                    "error_records": i,
                }

    trainer.train_loader = cast("DataLoader", Loader())
    trainer.rank = 0
    trainer.local_rank = "cpu"  # type: ignore[assignment]  # Exercise the loop on CPU.
    trainer.device_type = "cpu"
    trainer.is_distributed = False
    trainer.global_step = 0
    trainer.optimizers = [torch.optim.SGD(model.parameters(), lr=0.01)]
    trainer.schedulers = []
    callbacks = []
    trainer.train_epoch(0, step_callback=lambda: callbacks.append(trainer.global_step))
    assert model.forwards == updates * accum
    assert len(syncs) == updates
    assert trainer.global_step == updates
    assert callbacks == list(range(1, updates + 1))
    assert len(logged) == updates

    reference_weight = nn.Parameter(torch.tensor(0.25))
    reference_optimizer = torch.optim.SGD([reference_weight], lr=0.01)
    for update, record in enumerate(logged):
        values = torch.arange(update * accum + 1, (update + 1) * accum + 1).float()
        reference_optimizer.zero_grad()
        (reference_weight * (values / 10)).square().mean().backward()
        reference_optimizer.step()
        assert record["train"]["accuracy"] == pytest.approx(
            (values.sum() / values.square().sum()).item()
        )
        assert record["train"]["plain_metric"] == pytest.approx(values.mean().item())
        assert record["train"]["error_records"] == pytest.approx(values.mean().item())
    torch.testing.assert_close(model.weight, reference_weight)


def test_trainer_config_preserves_positional_resume():
    """Adding accumulation must preserve existing positional resume callers."""
    config = TrainerConfig(0.01, 1, "unused", True)
    assert config.resume_from_checkpoint is True
    assert config.gradient_accumulation_steps == 1
