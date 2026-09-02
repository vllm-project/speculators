from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import get_linear_schedule_with_warmup

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.config import TrainConfig
from speculators.train.trainer import (
    Trainer,
    TrainerConfig,
    WSDDecayStyle,
    _get_wsd_schedule_with_warmup,
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


def test_wsd_scheduler_via_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    optimizers = [
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=1.0),
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=0.1),
    ]
    monkeypatch.setattr(
        "speculators.train.trainer.build_optimizers",
        lambda _model, _config: optimizers,
    )

    trainer = Trainer.__new__(Trainer)
    trainer.model = cast("Any", torch.nn.Linear(2, 2))
    trainer.config = TrainerConfig(
        lr=1.0,
        num_epochs=1,
        save_path=str(tmp_path),
        scheduler_type="wsd",
        scheduler_warmup_steps=2,
        scheduler_total_steps=10,
        scheduler_warmup_init_lr_ratio=0.25,
        scheduler_min_lr_ratio=0.1,
        scheduler_wsd_decay_ratio=0.4,
        scheduler_wsd_decay_style="linear",
    )
    trainer.resume_from_checkpoint = False
    trainer.checkpointer = SingleGPUCheckpointer(tmp_path)
    trainer.train_loader = cast("Any", [None] * 10)

    trainer.setup_optimizer()

    observed = [[scheduler.get_last_lr()[0]] for scheduler in trainer.schedulers]
    for _ in range(10):
        for index, (optimizer, scheduler) in enumerate(
            zip(trainer.optimizers, trainer.schedulers, strict=True)
        ):
            optimizer.step()
            scheduler.step()
            observed[index].append(scheduler.get_last_lr()[0])

    expected = [
        0.25,
        0.625,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.775,
        0.55,
        0.325,
        0.1,
    ]
    assert len(trainer.schedulers) == 2
    assert observed[0] == pytest.approx(expected)
    assert observed[1] == pytest.approx([lr * 0.1 for lr in expected])


@pytest.mark.parametrize(
    ("decay_style", "expected_midpoint"),
    [
        ("linear", 0.5),
        ("cosine", 0.5),
        ("exponential", 2.0 * 0.5**0.5 - 1.0),
        ("minus_sqrt", 1.0 - 0.5**0.5),
    ],
)
def test_wsd_scheduler_supports_final_decay_styles(
    decay_style: WSDDecayStyle,
    expected_midpoint: float,
):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _get_wsd_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=4,
        decay_ratio=1.0,
        decay_style=decay_style,
    )

    for _ in range(2):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(expected_midpoint)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_training_steps": 0}, "num_training_steps"),
        ({"num_warmup_steps": -1}, "num_warmup_steps"),
        ({"num_warmup_steps": 10}, "num_warmup_steps"),
        ({"num_warmup_steps": 9, "decay_ratio": 0.2}, "overlap"),
        ({"warmup_init_lr_ratio": 1.1}, "warmup_init_lr_ratio"),
        ({"min_lr_ratio": -0.1}, "min_lr_ratio"),
        ({"decay_ratio": 0.0}, "decay_ratio"),
        ({"decay_style": "unknown"}, "decay_style"),
    ],
)
def test_wsd_scheduler_rejects_invalid_phase_geometry(
    overrides: dict[str, Any], message: str
):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    options: dict[str, Any] = {
        "num_warmup_steps": 2,
        "num_training_steps": 10,
        "decay_ratio": 0.2,
    }
    options.update(overrides)

    with pytest.raises(ValueError, match=message):
        _get_wsd_schedule_with_warmup(optimizer, **options)


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
