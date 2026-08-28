from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from transformers import get_linear_schedule_with_warmup

from speculators.train import checkpointer as checkpointer_module
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.config import TrainConfig
from speculators.train.optimizers import build_optimizers
from speculators.train.trainer import (
    Trainer,
    TrainerConfig,
    _resolve_scheduler_steps,
)


def make_config(**overrides) -> TrainerConfig:
    return TrainerConfig(
        lr=1e-4,
        num_epochs=5,
        save_path="checkpoint",
        **overrides,
    )


def test_scheduler_steps_default_to_one_percent_of_training_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(make_config(), 20)

    assert total_steps == 100
    assert warmup_steps == 1


def test_scheduler_total_steps_only_defaults_warmup_to_one_percent_of_total():
    # default_total_steps is num_epochs * loader_len = 100, but the explicit
    # scheduler_total_steps override must drive the 1% warmup fallback (10, not 1).
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(scheduler_total_steps=1000),
        20,
    )

    assert total_steps == 1000
    assert warmup_steps == 10


def test_scheduler_warmup_ratio_uses_scheduler_total_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(scheduler_total_steps=200, scheduler_warmup_ratio=0.1),
        20,
    )

    assert total_steps == 200
    assert warmup_steps == 20


def test_scheduler_warmup_steps_take_precedence_over_ratio():
    with pytest.warns(UserWarning, match="using scheduler_warmup_steps"):
        warmup_steps, total_steps = _resolve_scheduler_steps(
            make_config(scheduler_warmup_steps=0, scheduler_warmup_ratio=0.1),
            20,
        )

    assert total_steps == 100
    assert warmup_steps == 0


def test_scheduler_warmup_ratio_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="scheduler_warmup_ratio"):
        _resolve_scheduler_steps(make_config(scheduler_warmup_ratio=1.1), 20)


def test_scheduler_type_rejects_unsupported_values():
    # --verifier-name-or-path is supplied so the only parse failure is the rejected
    # --scheduler-type choice (not the missing required verifier arg).
    with pytest.raises(SystemExit):
        TrainConfig.resolve(
            ["--verifier-name-or-path", "x", "--scheduler-type", "constant"]
        )


def test_scheduler_resume_restores_optimizer_learning_rate(tmp_path: Path):
    checkpoint_dir = tmp_path / "0"
    checkpoint_dir.mkdir()
    checkpointer = SingleGPUCheckpointer(tmp_path)

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )
    for _ in range(25):
        optimizer.step()
        scheduler.step()

    expected_lr = scheduler.get_last_lr()[0]
    checkpointer.save_scheduler_state_dict(scheduler, epoch=0)

    resumed_parameter = torch.nn.Parameter(torch.zeros(()))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-3)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scheduler = get_linear_schedule_with_warmup(
        resumed_optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        last_epoch=0,
    )
    assert resumed_optimizer.param_groups[0]["lr"] != pytest.approx(expected_lr)

    checkpointer.load_scheduler_state_dict(resumed_scheduler)

    assert resumed_scheduler.get_last_lr()[0] == pytest.approx(expected_lr)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_resume_without_scheduler_state_keeps_configured_base_lr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A checkpoint may carry a decayed optimizer LR but no scheduler file
    (legacy checkpoints, or an interruption between the optimizer and scheduler
    writes). The rebuilt scheduler must adopt the configured LR as its base,
    not the decayed value the optimizer load put back into the param groups."""
    monkeypatch.setattr(checkpointer_module, "get_current_device", lambda: "cpu")

    config = TrainerConfig(
        lr=1e-4,
        num_epochs=5,
        save_path=str(tmp_path),
        resume_from_checkpoint=True,
    )
    model = torch.nn.Linear(2, 2)
    model.dtype = torch.float32  # load_optimizer_state_dict reads it off the model

    # Checkpoint an optimizer whose scheduled LR has decayed to ~zero.
    (tmp_path / "0").mkdir()
    saved = build_optimizers(model, config)
    for group in saved[0].param_groups:
        group["lr"] = 1e-9
    torch.save([saved[0].state_dict()], tmp_path / "0" / "optimizer_state_dict.pt")

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.config = config
    trainer.resume_from_checkpoint = True
    trainer.checkpointer = SingleGPUCheckpointer(tmp_path)
    trainer.train_loader = MagicMock(__len__=MagicMock(return_value=20))
    trainer.setup_optimizer()

    assert trainer.schedulers[0].base_lrs == [pytest.approx(config.lr)]
    assert trainer.optimizers[0].param_groups[0]["lr"] != pytest.approx(1e-9)
