from pathlib import Path

import pytest
import torch
from transformers import get_linear_schedule_with_warmup

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.config import TrainConfig
from speculators.train.trainer import (
    Trainer,
    TrainerConfig,
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


def test_wsd_scheduler_options_resolve_from_cli():
    flat = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "x",
            "--scheduler-type",
            "wsd",
            "--scheduler-warmup-init-lr-ratio",
            "0.25",
            "--scheduler-min-lr-ratio",
            "0.1",
            "--scheduler-wsd-decay-ratio",
            "0.2",
            "--scheduler-wsd-decay-style",
            "linear",
        ]
    ).flatten()

    assert flat["scheduler_type"] == "wsd"
    assert flat["scheduler_warmup_init_lr_ratio"] == pytest.approx(0.25)
    assert flat["scheduler_min_lr_ratio"] == pytest.approx(0.1)
    assert flat["scheduler_wsd_decay_ratio"] == pytest.approx(0.2)
    assert flat["scheduler_wsd_decay_style"] == "linear"


def test_wsd_scheduler_has_warmup_stable_and_final_cosine_decay(tmp_path: Path):
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.config = TrainerConfig(
        lr=1.0,
        num_epochs=1,
        save_path=str(tmp_path),
        scheduler_type="wsd",
        scheduler_warmup_steps=2,
        scheduler_total_steps=10,
    )
    trainer.resume_from_checkpoint = False
    trainer.checkpointer = SingleGPUCheckpointer(tmp_path)
    trainer.train_loader = [None] * 10

    trainer.setup_optimizer()

    scheduler = trainer.schedulers[0]
    observed = [scheduler.get_last_lr()[0]]
    for _ in range(10):
        trainer.optimizers[0].step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])

    assert observed == pytest.approx(
        [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
    )


def test_wsd_scheduler_honors_init_floor_and_decay_ratio():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _get_wsd_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        warmup_init_lr_ratio=0.25,
        min_lr_ratio=0.1,
        decay_ratio=0.4,
        decay_style="linear",
    )

    observed = [scheduler.get_last_lr()[0]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])

    assert observed == pytest.approx(
        [0.25, 0.625, 1.0, 1.0, 1.0, 1.0, 1.0, 0.775, 0.55, 0.325, 0.1]
    )


def test_wsd_scheduler_scales_each_optimizer_group_relative_to_its_peak_lr():
    first = torch.nn.Parameter(torch.zeros(()))
    second = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 1.0},
            {"params": [second], "lr": 0.1},
        ]
    )
    scheduler = _get_wsd_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=2,
        min_lr_ratio=0.1,
        decay_ratio=1.0,
        decay_style="linear",
    )

    assert scheduler.get_last_lr() == pytest.approx([1.0, 0.1])
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.1, 0.01])


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
    decay_style: str,
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
def test_wsd_scheduler_rejects_invalid_phase_geometry(overrides, message):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    options = {
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
