import pytest

from scripts.train import parse_args
from speculators.train.trainer import (
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


def test_scheduler_type_rejects_unsupported_values(monkeypatch):
    # --verifier-name-or-path is supplied so the only parse failure is the rejected
    # --scheduler-type choice (not the missing required verifier arg).
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--verifier-name-or-path", "x", "--scheduler-type", "constant"],
    )

    with pytest.raises(SystemExit):
        parse_args()
