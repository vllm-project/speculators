"""Unit test: scheduler resume seeds last_epoch in optimizer steps, not epochs.

The warmup schedulers step once per optimizer step. Seeding them with the epoch
index on resume (when scheduler_state_dict.pt is absent) restarts the LR
schedule from near zero instead of continuing from the resumed global step.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from speculators.train.trainer import Trainer

RESUMED_GLOBAL_STEP = 500


class _FakeCheckpointer:
    previous_epoch = 2

    def load_optimizer_state_dict(self, model, optimizers):
        pass

    def scheduler_path(self, epoch):
        return Path(f"/nonexistent/{epoch}/scheduler_state_dict.pt")

    def load_scheduler_state_dict(self, schedulers):
        # Mirrors BaseCheckpointer when scheduler_state_dict.pt is missing.
        pass


def _make_trainer() -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.model = nn.Linear(4, 4)
    trainer.config = SimpleNamespace(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=0.0,
        scheduler_type="linear",
        scheduler_total_steps=1000,
        scheduler_warmup_steps=100,
        scheduler_warmup_ratio=None,
        num_epochs=4,
    )
    trainer.train_loader = list(range(250))
    trainer.resume_from_checkpoint = True
    trainer.checkpointer = _FakeCheckpointer()
    trainer.global_step = RESUMED_GLOBAL_STEP
    return trainer


def test_scheduler_resume_seed_uses_global_step():
    trainer = _make_trainer()
    trainer.setup_optimizer()
    (scheduler,) = trainer.schedulers
    assert scheduler.last_epoch == RESUMED_GLOBAL_STEP

    # The resumed LR must equal a fresh scheduler advanced by the same number
    # of optimizer steps, not the LR at step previous_epoch + 1 of warmup.
    reference_model = nn.Linear(4, 4)
    reference_opt = torch.optim.AdamW(reference_model.parameters(), lr=1e-3)
    reference = get_linear_schedule_with_warmup(
        reference_opt, num_warmup_steps=100, num_training_steps=1000
    )
    for _ in range(RESUMED_GLOBAL_STEP):
        reference.step()
    assert scheduler.get_last_lr() == reference.get_last_lr()


def test_scheduler_fresh_run_still_starts_at_minus_one():
    trainer = _make_trainer()
    trainer.resume_from_checkpoint = False
    trainer.global_step = 0
    trainer.setup_optimizer()
    (scheduler,) = trainer.schedulers
    assert scheduler.last_epoch == 0  # constructor's initial step from -1
