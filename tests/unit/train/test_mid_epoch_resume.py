"""Unit tests for mid-epoch checkpoint save and resume.

Verifies:
1. training_state.json is saved alongside each fractional checkpoint
2. On resume, epoch is correctly restored (not bumped to next epoch)
3. local_step skip works — no duplicate or missing batches
4. global_step continues from checkpoint value
5. end-of-epoch checkpoint (local_step=0) causes resume to advance epoch
6. 'interrupted' checkpoint does not write training_state.json
7. Descriptive symlinks are created and updated correctly
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

trained_steps: list[tuple[int, int, int]] = []


def _patch_checkpointer() -> None:
    """Stub out all checkpointer I/O that requires real model weights."""
    SingleGPUCheckpointer.save_checkpoint = (  # type: ignore[method-assign]
        lambda self, model, opt, epoch: (self.path / str(epoch)).mkdir(
            parents=True, exist_ok=True
        )
    )
    SingleGPUCheckpointer.save_scheduler_state_dict = lambda *a: None  # type: ignore[method-assign]
    SingleGPUCheckpointer.load_model_state_dict = lambda *a: None  # type: ignore[method-assign]
    SingleGPUCheckpointer.load_optimizer_state_dict = lambda *a: None  # type: ignore[method-assign]
    SingleGPUCheckpointer.load_scheduler_state_dict = lambda *a: None  # type: ignore[method-assign]


_patch_checkpointer()


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": torch.tensor([i]), "loss_mask": torch.tensor([1.0])}


def _make_loader() -> DataLoader:
    return DataLoader(_TinyDataset(), batch_size=10, shuffle=False)


class _MockTrainer(Trainer):
    """Trainer subclass that records steps without GPU/model ops."""

    def setup_model(self) -> None:
        pass

    def setup_optimizer(self) -> None:
        p = nn.Parameter(torch.zeros(1))
        self.opt = torch.optim.SGD([p], lr=1e-4)
        self.scheduler = None

    def train_epoch(self, epoch: int) -> None:
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)

        skip_steps = 0
        if epoch == getattr(self, "current_epoch", epoch):
            skip_steps = getattr(self, "_resume_local_step", 0)
            self._resume_local_step = 0

        num_steps = len(self.train_loader)
        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )

        for local_step, _batch in enumerate(self.train_loader, 1):
            if local_step <= skip_steps:
                continue
            trained_steps.append((epoch, local_step, self.global_step))
            self.global_step += 1
            if (
                step_interval
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * 0.1
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)


def _make_trainer(save_path: str, resume: bool = False, epochs: int = 1) -> _MockTrainer:
    cfg = TrainerConfig(
        save_path=save_path,
        num_epochs=epochs,
        lr=1e-4,
        local_rank=0,
        is_distributed=False,
        resume_from_checkpoint=resume,
        checkpoint_freq=0.3,
        log_freq=1,
        scheduler_type="none",
    )
    return _MockTrainer(cfg, cfg, _make_loader())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mid_epoch_checkpoint_saves_training_state() -> None:
    """training_state.json is written with correct epoch/local_step/global_step."""
    trained_steps.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))
        t = _make_trainer(tmpdir)
        for local_step, _batch in enumerate(t.train_loader, 1):
            trained_steps.append((0, local_step, t.global_step))
            t.global_step += 1
            if local_step == step_interval:
                t.maybe_save_checkpoint(0, local_step=local_step)
                break

        state_file = Path(tmpdir) / "0" / "training_state.json"
        assert state_file.exists(), "training_state.json was not saved"
        state = json.loads(state_file.read_text())
        assert state == {"epoch": 0, "local_step": step_interval, "global_step": step_interval}


def test_mid_epoch_resume_restores_epoch_and_step() -> None:
    """Resume from mid-epoch checkpoint stays in same epoch and skips correct batches."""
    trained_steps.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))

        # Run 1: interrupt after first checkpoint
        t1 = _make_trainer(tmpdir)
        for local_step, _batch in enumerate(t1.train_loader, 1):
            trained_steps.append((0, local_step, t1.global_step))
            t1.global_step += 1
            if local_step == step_interval:
                t1.maybe_save_checkpoint(0, local_step=local_step)
                break

        # Run 2: resume
        trained_steps.clear()
        t2 = _make_trainer(tmpdir, resume=True)
        assert t2.current_epoch == 0, f"Expected epoch 0, got {t2.current_epoch}"
        assert t2._resume_local_step == step_interval
        assert t2.global_step == step_interval

        t2.train_epoch(t2.current_epoch)

        expected = num_steps - step_interval
        assert len(trained_steps) == expected
        assert trained_steps[0][1] == step_interval + 1  # first local_step after skip
        assert trained_steps[0][2] == step_interval  # global_step continues
        assert trained_steps[-1][1] == num_steps


def test_end_of_epoch_checkpoint_advances_epoch() -> None:
    """End-of-epoch checkpoint (local_step=0) causes resume to advance to next epoch."""
    trained_steps.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, epochs=2)
        t.train_epoch(0)
        t.maybe_save_checkpoint(0, local_step=0)

        trained_steps.clear()
        t2 = _make_trainer(tmpdir, resume=True, epochs=2)
        assert t2.current_epoch == 1, f"Expected epoch 1, got {t2.current_epoch}"
        assert t2._resume_local_step == 0


def test_interrupted_checkpoint_has_no_training_state() -> None:
    """'interrupted' checkpoint does not write training_state.json."""
    trained_steps.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir)
        t.maybe_save_checkpoint("interrupted")
        state_file = Path(tmpdir) / "interrupted" / "training_state.json"
        assert not state_file.exists()


def test_symlink_created_and_updated() -> None:
    """Symlink is created for mid-epoch and updated when overwritten."""
    trained_steps.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))
        t = _make_trainer(tmpdir)
        t.maybe_save_checkpoint(0, local_step=step_interval)
        t.maybe_save_checkpoint(0, local_step=step_interval * 2)

        old_link = Path(tmpdir) / f"epoch0_step{step_interval}"
        new_link = Path(tmpdir) / f"epoch0_step{step_interval * 2}"
        assert not old_link.exists(), "old symlink should be removed"
        assert new_link.is_symlink(), "new symlink should exist"

        state = json.loads((Path(tmpdir) / "0" / "training_state.json").read_text())
        assert state["local_step"] == step_interval * 2


def test_end_of_epoch_symlink() -> None:
    """End-of-epoch checkpoint creates epoch{N}_end symlink."""
    trained_steps.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir)
        t.maybe_save_checkpoint(0, local_step=0)
        end_link = Path(tmpdir) / "epoch0_end"
        assert end_link.is_symlink()
