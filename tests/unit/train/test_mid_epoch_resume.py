"""Unit tests for mid-epoch checkpoint save and resume."""

import json
import tempfile
from pathlib import Path
from typing import Protocol, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_steps() -> list[tuple[int, int, int]]:
    """Per-test collection of (epoch, local_step, global_step) tuples."""
    return []


@pytest.fixture(autouse=True)
def patch_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub checkpointer I/O that requires real model weights or process groups."""

    def _save_checkpoint(self: object, *args: object, **kwargs: object) -> None:
        epoch = args[2] if len(args) >= 3 else kwargs.get("epoch", "0")
        self.path.joinpath(str(epoch)).mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    for cls in (SingleGPUCheckpointer, DistributedCheckpointer):
        monkeypatch.setattr(cls, "save_checkpoint", _save_checkpoint)
        monkeypatch.setattr(cls, "save_scheduler_state_dict", _noop)
        monkeypatch.setattr(cls, "load_model_state_dict", _noop)
        monkeypatch.setattr(cls, "load_optimizer_state_dict", _noop)
        monkeypatch.setattr(cls, "load_scheduler_state_dict", _noop)


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": torch.tensor([i]), "loss_mask": torch.tensor([1.0])}


def _make_loader() -> DataLoader:
    return DataLoader(_TinyDataset(), batch_size=10, shuffle=False)


class _BatchSamplerWithSetEpoch(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class _FastSkipBatchSamplerProtocol(Protocol):
    _cached_generated_batches: tuple[int, list[list[int]]] | None

    def _generate_batches(self, epoch: int) -> list[list[int]]: ...


def _dummy_model() -> SpeculatorModel:
    return cast("SpeculatorModel", nn.Identity())


class _MockTrainer(Trainer):
    """Trainer subclass that records steps without GPU/model ops."""

    _trained_steps: list[tuple[int, int, int]]

    def setup_model(self) -> None:
        pass

    def setup_optimizer(self) -> None:
        p = nn.Parameter(torch.zeros(1))
        self.opt = torch.optim.AdamW([p], lr=1e-4)
        self.scheduler = None

    def train_epoch(self, epoch: int) -> None:
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            batch_sampler = cast(
                "_BatchSamplerWithSetEpoch", self.train_loader.batch_sampler
            )
            batch_sampler.set_epoch(epoch)

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
            self._trained_steps.append((epoch, local_step, self.global_step))
            self.global_step += 1
            if (
                step_interval
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * 0.1
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)


def _make_trainer(
    save_path: str,
    trained_steps: list[tuple[int, int, int]],
    resume: bool = False,
    epochs: int = 1,
    is_distributed: bool = False,
) -> _MockTrainer:
    cfg = TrainerConfig(
        save_path=save_path,
        num_epochs=epochs,
        lr=1e-4,
        local_rank=0,
        is_distributed=is_distributed,
        resume_from_checkpoint=resume,
        checkpoint_freq=0.3,
        log_freq=1,
        scheduler_type="none",
    )
    trainer = _MockTrainer(_dummy_model(), cfg, _make_loader())
    trainer._trained_steps = trained_steps
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mid_epoch_checkpoint_saves_training_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """training_state.json is written with correct epoch/local_step/global_step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        for local_step, _batch in enumerate(t.train_loader, 1):
            trained_steps.append((0, local_step, t.global_step))
            t.global_step += 1
            if local_step == step_interval:
                t.maybe_save_checkpoint(0, local_step=local_step)
                break

        state_file = Path(tmpdir) / "0" / "training_state.json"
        assert state_file.exists(), "training_state.json was not saved"
        state = json.loads(state_file.read_text())
        expected = {
            "epoch": 0,
            "local_step": step_interval,
            "global_step": step_interval,
        }
        assert state == expected


def test_mid_epoch_resume_restores_epoch_and_step(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Resume from mid-epoch checkpoint stays in same epoch and skips batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_fraction = round(num_steps * 0.3)
        step_interval = max(1, step_fraction)

        # Run 1: interrupt after first checkpoint.
        run1_steps = trained_steps
        t1 = _make_trainer(tmpdir, trained_steps=run1_steps)
        for local_step, _batch in enumerate(t1.train_loader, 1):
            run1_steps.append((0, local_step, t1.global_step))
            t1.global_step += 1
            if local_step == step_interval:
                t1.maybe_save_checkpoint(0, local_step=local_step)
                break

        # Run 2: resume.
        run2_steps: list[tuple[int, int, int]] = []
        t2 = _make_trainer(tmpdir, trained_steps=run2_steps, resume=True)
        assert t2.current_epoch == 0, f"Expected epoch 0, got {t2.current_epoch}"
        assert t2._resume_local_step == step_interval
        assert t2.global_step == step_interval

        t2.train_epoch(t2.current_epoch)

        expected = num_steps - step_interval
        assert len(run2_steps) == expected
        assert run2_steps[0][1] == step_interval + 1  # first local_step after skip
        assert run2_steps[0][2] == step_interval  # global_step continues
        assert run2_steps[-1][1] == num_steps


def test_end_of_epoch_checkpoint_advances_epoch(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """End-of-epoch checkpoint (local_step=0) resumes at next epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps, epochs=2)
        t.train_epoch(0)
        t.maybe_save_checkpoint(0, local_step=0)

        run2_steps: list[tuple[int, int, int]] = []
        t2 = _make_trainer(tmpdir, trained_steps=run2_steps, resume=True, epochs=2)
        assert t2.current_epoch == 1, f"Expected epoch 1, got {t2.current_epoch}"
        assert t2._resume_local_step == 0


def test_interrupted_checkpoint_has_no_training_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """'interrupted' checkpoint does not write training_state.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint("interrupted")
        state_file = Path(tmpdir) / "interrupted" / "training_state.json"
        assert not state_file.exists()


def test_symlink_created_and_updated(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Symlink is created for mid-epoch and updated when overwritten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint(0, local_step=step_interval)
        t.maybe_save_checkpoint(0, local_step=step_interval * 2)

        old_link = Path(tmpdir) / f"epoch0_step{step_interval}"
        new_link = Path(tmpdir) / f"epoch0_step{step_interval * 2}"
        assert not old_link.exists(), "old symlink should be removed"
        assert new_link.is_symlink(), "new symlink should exist"

        state = json.loads((Path(tmpdir) / "0" / "training_state.json").read_text())
        assert state["local_step"] == step_interval * 2


def test_end_of_epoch_symlink(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """End-of-epoch checkpoint creates epoch{N}_end symlink."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint(0, local_step=0)
        end_link = Path(tmpdir) / "epoch0_end"
        assert end_link.is_symlink()


def test_distributed_mid_epoch_checkpoint_rank_gate(
    trained_steps: list[tuple[int, int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-0 writes state/symlink while nonzero ranks skip side effects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_steps = len(_make_loader())
        step_interval = max(1, round(num_steps * 0.3))

        rank0 = _make_trainer(
            tmpdir,
            trained_steps=trained_steps,
            is_distributed=True,
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        rank0.maybe_save_checkpoint(0, local_step=step_interval)

        state_rank0 = Path(tmpdir) / "0" / "training_state.json"
        link_rank0 = Path(tmpdir) / f"epoch0_step{step_interval}"
        assert state_rank0.exists()
        assert link_rank0.is_symlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        rank1_steps: list[tuple[int, int, int]] = []
        rank1 = _make_trainer(
            tmpdir,
            trained_steps=rank1_steps,
            is_distributed=True,
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
        rank1.maybe_save_checkpoint(0, local_step=step_interval)

        state_rank1 = Path(tmpdir) / "0" / "training_state.json"
        link_rank1 = Path(tmpdir) / f"epoch0_step{step_interval}"
        assert not state_rank1.exists()
        assert not link_rank1.exists()


class _CountingDataset(Dataset):
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.seen_indices: list[int] = []

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> dict:
        self.seen_indices.append(idx)
        return {
            "input_ids": torch.tensor([idx]),
            "loss_mask": torch.tensor([1.0]),
        }


class _FastSkipBatchSampler:
    def __init__(self, n_items: int):
        self.all_batches = [[i] for i in range(n_items)]
        self._cached_generated_batches: tuple[int, list[list[int]]] | None = None
        self.generated_for_epoch: int | None = None
        self.current_epoch = 0

    def __len__(self) -> int:
        if self._cached_generated_batches is not None:
            return len(self._cached_generated_batches[1])
        return len(self.all_batches)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def _generate_batches(self, epoch: int) -> list[list[int]]:
        self.generated_for_epoch = epoch
        return list(self.all_batches)

    def __iter__(self):
        if (
            self._cached_generated_batches is not None
            and self._cached_generated_batches[0] == self.current_epoch
        ):
            yield from self._cached_generated_batches[1]
            return
        yield from self._generate_batches(self.current_epoch)


class _FastSkipMockTrainer(_MockTrainer):
    def train_epoch(self, epoch: int) -> None:
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            batch_sampler = cast(
                "_BatchSamplerWithSetEpoch", self.train_loader.batch_sampler
            )
            batch_sampler.set_epoch(epoch)

        skip_steps = 0
        if epoch == getattr(self, "current_epoch", epoch):
            skip_steps = getattr(self, "_resume_local_step", 0)
            self._resume_local_step = 0

        sampler = self.train_loader.batch_sampler
        has_fast_skip_api = hasattr(sampler, "_generate_batches") and hasattr(
            sampler, "_cached_generated_batches"
        )
        if skip_steps > 0 and has_fast_skip_api:
            fast_skip_sampler = cast("_FastSkipBatchSamplerProtocol", sampler)
            all_batches = fast_skip_sampler._generate_batches(epoch)
            remaining = all_batches[skip_steps:]
            fast_skip_sampler._cached_generated_batches = (epoch, remaining)

        for local_step_rel, _batch in enumerate(self.train_loader, 1):
            local_step = local_step_rel + skip_steps
            self._trained_steps.append((epoch, local_step, self.global_step))
            self.global_step += 1


def test_fast_skip_sampler_slice_avoids_skipped_getitem(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Fast-skip avoids __getitem__ calls for skipped batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _CountingDataset(n_items=10)
        sampler = _FastSkipBatchSampler(n_items=10)
        loader = DataLoader(dataset, batch_sampler=sampler)
        cfg = TrainerConfig(
            save_path=tmpdir,
            num_epochs=1,
            lr=1e-4,
            local_rank=0,
            is_distributed=False,
            resume_from_checkpoint=False,
            checkpoint_freq=0.3,
            log_freq=1,
            scheduler_type="none",
        )
        trainer = _FastSkipMockTrainer(_dummy_model(), cfg, loader)
        trainer._trained_steps = trained_steps
        trainer._resume_local_step = 3

        trainer.train_epoch(0)

        assert sampler.generated_for_epoch == 0
        assert sampler._cached_generated_batches == (0, sampler.all_batches[3:])
        assert dataset.seen_indices == list(range(3, 10))
