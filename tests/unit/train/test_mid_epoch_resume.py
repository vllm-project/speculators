"""Unit tests for mid-epoch checkpoint save and resume."""

import json
import random
import tempfile
from pathlib import Path
from typing import Protocol, cast

import numpy as np
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
        opt = torch.optim.AdamW([p], lr=1e-4)
        self.opt = opt
        self.optimizers = [opt]
        self.scheduler = None
        self.schedulers = []

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

        for local_step, _batch in enumerate(self._epoch_iterator(skip_steps), 1):
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
) -> _MockTrainer:
    cfg = TrainerConfig(
        save_path=save_path,
        num_epochs=epochs,
        lr=1e-4,
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
        )
        rank0.is_distributed = True
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
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
        )
        rank1.is_distributed = True
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
        rank1.maybe_save_checkpoint(0, local_step=step_interval)

        state_rank1 = Path(tmpdir) / "0" / "training_state.json"
        link_rank1 = Path(tmpdir) / f"epoch0_step{step_interval}"
        assert not state_rank1.exists()
        assert not link_rank1.exists()
        # RNG state is per rank: every rank writes its own file.
        assert (Path(tmpdir) / "0" / "rng_state_rank1.pt").exists()


def _draw_all() -> tuple[torch.Tensor, float, float]:
    return torch.rand(4), random.random(), np.random.rand()  # noqa: NPY002


class _NoisyDataset(Dataset):
    """Every fetch draws from the CPU generator, like a noise transform does."""

    def __init__(self, n_items: int = 10):
        self.n_items = n_items

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": torch.tensor([idx]), "noise": torch.rand(3)}


def _noisy_fast_skip_loader(n_items: int = 10) -> DataLoader:
    return DataLoader(
        _NoisyDataset(n_items), batch_sampler=_FastSkipBatchSampler(n_items)
    )


def _make_trainer_with_loader(
    save_path: str, loader: DataLoader, resume: bool = False, epochs: int = 1
) -> _MockTrainer:
    cfg = TrainerConfig(
        save_path=save_path,
        num_epochs=epochs,
        lr=1e-4,
        resume_from_checkpoint=resume,
        checkpoint_freq=0.3,
        log_freq=1,
        scheduler_type="none",
    )
    trainer = _MockTrainer(_dummy_model(), cfg, loader)
    trainer._trained_steps = []
    return trainer


def test_mid_epoch_checkpoint_saves_rng_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Mid-epoch checkpoints store a weights_only-loadable RNG snapshot at save time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint(0, local_step=3)
        rng_file = Path(tmpdir) / "0" / "rng_state_rank0.pt"
        assert rng_file.exists()
        state = torch.load(rng_file, map_location="cpu", weights_only=True)
        assert set(state) >= {"python", "numpy", "torch_cpu", "device_type"}
        assert torch.equal(state["torch_cpu"], torch.get_rng_state())

        t.maybe_save_checkpoint("interrupted")
        assert not (Path(tmpdir) / "interrupted" / "rng_state_rank0.pt").exists()


def test_end_of_epoch_rng_state_is_written_after_validation(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """End-of-epoch checkpoints snapshot RNG at the end of the epoch loop, not."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t = _make_trainer(tmpdir, trained_steps=trained_steps)
        t.maybe_save_checkpoint(0, local_step=0)
        rng_file = Path(tmpdir) / "0" / "rng_state_rank0.pt"
        assert not rng_file.exists()  # validation has not run yet
        torch.rand(2)  # whatever validation consumes
        t._save_end_of_epoch_rng_state(0)
        assert rng_file.exists()
        state = torch.load(rng_file, map_location="cpu", weights_only=True)
        assert torch.equal(state["torch_cpu"], torch.get_rng_state())
        # Idempotent: no checkpoint written for epoch 1 -> nothing to snapshot.
        t._save_end_of_epoch_rng_state(1)
        assert not (Path(tmpdir) / "1" / "rng_state_rank0.pt").exists()


def test_save_best_checkpoint_gets_rng_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """save_best=True checkpoints (from maybe_update_best) get the snapshot too."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainerConfig(
            save_path=tmpdir,
            num_epochs=1,
            lr=1e-4,
            resume_from_checkpoint=False,
            checkpoint_freq=1,
            save_best=True,
            log_freq=1,
            scheduler_type="none",
        )
        t = _MockTrainer(_dummy_model(), cfg, _make_loader())
        t._trained_steps = trained_steps
        t.maybe_save_checkpoint(0)  # returns early under save_best
        assert not (Path(tmpdir) / "0").exists()
        t.maybe_update_best(0, {"loss_epoch": 0.5})
        t._save_end_of_epoch_rng_state(0)
        assert (Path(tmpdir) / "0" / "rng_state_rank0.pt").exists()


def test_mid_epoch_resume_continues_the_interrupted_iterator_exactly() -> None:
    """A new process resuming mid-epoch draws what the live iterator would have drawn.

    The uninterrupted run keeps its existing DataLoader iterator after the
    checkpoint; the resumed run must create a new one (which draws a base seed
    from the CPU generator) *before* the snapshot is applied, or the two diverge.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Uninterrupted run: iterator alive, one step done, checkpoint, continue.
        t1 = _make_trainer_with_loader(tmpdir, _noisy_fast_skip_loader())
        torch.manual_seed(1234)
        random.seed(5)
        np.random.seed(7)  # noqa: NPY002
        it1 = t1._epoch_iterator(t1._prepare_resume_skip(0))
        next(it1)
        t1.global_step = 1
        t1.maybe_save_checkpoint(0, local_step=1)
        expected_batch = next(it1)  # the live iterator fetches step 2
        expected = _draw_all()

        # Perturb every generator, then resume in a fresh trainer/loader.
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)  # noqa: NPY002
        t2 = _make_trainer_with_loader(tmpdir, _noisy_fast_skip_loader(), resume=True)
        assert t2.current_epoch == 0
        assert t2._pending_rng_state is not None
        skip = t2._prepare_resume_skip(0)
        assert skip == 1
        it2 = t2._epoch_iterator(skip)
        assert t2._pending_rng_state is None
        got_batch = next(it2)  # first resumed fetch is step 2
        got = _draw_all()

        assert torch.equal(got_batch["input_ids"], expected_batch["input_ids"])
        assert torch.equal(got_batch["noise"], expected_batch["noise"])
        assert torch.equal(got[0], expected[0])
        assert got[1] == expected[1]
        assert got[2] == expected[2]


def test_end_of_epoch_resume_continues_the_next_epoch_exactly() -> None:
    """Resuming at an epoch boundary reproduces the next epoch's first fetch/draws."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t1 = _make_trainer_with_loader(tmpdir, _noisy_fast_skip_loader(), epochs=2)
        torch.manual_seed(99)
        t1.maybe_save_checkpoint(0, local_step=0)
        torch.rand(2)  # validation
        t1._save_end_of_epoch_rng_state(0)
        # Uninterrupted run starts epoch 1: new iterator (base seed draw) + fetch.
        t1.current_epoch = 1
        it1 = t1._epoch_iterator(t1._prepare_resume_skip(1))
        expected_batch = next(it1)
        expected = torch.rand(4)

        torch.manual_seed(0)
        t2 = _make_trainer_with_loader(
            tmpdir, _noisy_fast_skip_loader(), resume=True, epochs=2
        )
        assert t2.current_epoch == 1
        it2 = t2._epoch_iterator(t2._prepare_resume_skip(1))
        got_batch = next(it2)
        assert torch.equal(got_batch["noise"], expected_batch["noise"])
        assert torch.equal(torch.rand(4), expected)


def test_mock_train_epoch_consumes_pending_rng_state(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """train_epoch applies the pending snapshot exactly once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t1 = _make_trainer(tmpdir, trained_steps=trained_steps)
        t1.maybe_save_checkpoint(0, local_step=3)
        t2 = _make_trainer(tmpdir, trained_steps=[], resume=True)
        assert t2._pending_rng_state is not None
        t2.train_epoch(t2.current_epoch)
        assert t2._pending_rng_state is None


def test_resume_without_rng_state_file_still_works(
    trained_steps: list[tuple[int, int, int]],
) -> None:
    """Checkpoints written before this feature resume with a warning, not a crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t1 = _make_trainer(tmpdir, trained_steps=trained_steps)
        t1.maybe_save_checkpoint(0, local_step=3)
        (Path(tmpdir) / "0" / "rng_state_rank0.pt").unlink()

        t2 = _make_trainer(tmpdir, trained_steps=[], resume=True)
        assert t2._pending_rng_state is None
        t2._maybe_restore_rng_state()  # no-op


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

        for local_step_rel, _batch in enumerate(self._epoch_iterator(skip_steps), 1):
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
