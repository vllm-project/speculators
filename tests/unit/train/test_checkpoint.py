from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer, TrainerConfig


def _make_minimal_trainer(tmp_path: Path, checkpoint_freq: int, save_best: bool):
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=0,
        save_path=str(tmp_path),
        resume_from_checkpoint=False,
        is_distributed=False,
        local_rank=0,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
    )
    trainer.best_val_loss = float("inf")
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.is_distributed = False
    trainer.local_rank = 0
    trainer.resume_from_checkpoint = False
    trainer.train_loader = cast("DataLoader[Any]", [])
    trainer.val_loader = cast("DataLoader[Any]", [])
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))

    trainer.model = cast("SpeculatorModel", object())
    trainer.opt = cast("torch.optim.AdamW", object())
    trainer.scheduler = None
    return trainer


def test_previous_epoch_ignores_checkpoint_best(tmp_path: Path):
    (tmp_path / "0").mkdir()
    (tmp_path / "2").mkdir()
    (tmp_path / "checkpoint_best").symlink_to("0", target_is_directory=True)

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.previous_epoch == 2


def test_update_best_symlink_creates_and_updates(tmp_path: Path):
    (tmp_path / "1").mkdir()
    (tmp_path / "3").mkdir()

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.update_best_symlink(1)

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "1").resolve()

    cp.update_best_symlink(3)
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "3").resolve()


def test_run_training_updates_checkpoint_best_among_saved_checkpoints_save_best_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=2, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_epochs = []
    val_losses = {
        0: 0.9,
        1: 0.6,
        2: 0.1,
        3: 0.7,
    }

    def fake_train_epoch(epoch: int):
        return None

    def fake_val_epoch(epoch: int):
        return {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.train_epoch = fake_train_epoch
    trainer.val_epoch = fake_val_epoch
    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == [0, 1, 3]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "1").resolve()
    assert trainer.best_val_loss == 0.6


@pytest.mark.parametrize(
    (
        "save_best",
        "checkpoint_freq",
        "expected_saved",
        "expected_remaining_dirs",
        "expected_best_target",
    ),
    [
        (False, 3, [0, 2], {"0", "2"}, "2"),
        (True, 3, [0, 1, 3], {"3"}, "3"),
    ],
)
def test_save_best_flag_changes_checkpoint_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    save_best: bool,
    checkpoint_freq: int,
    expected_saved: list[int],
    expected_remaining_dirs: set[str],
    expected_best_target: str,
):
    case_dir = tmp_path / ("save_best" if save_best else "save_freq")
    case_dir.mkdir()
    trainer = _make_minimal_trainer(
        case_dir, checkpoint_freq=checkpoint_freq, save_best=save_best
    )
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_epochs: list[int] = []
    val_losses = {0: 0.9, 1: 0.8, 2: 0.85, 3: 0.7}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (case_dir / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == expected_saved

    remaining_dirs = {
        p.name for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit()
    }
    assert remaining_dirs == expected_remaining_dirs

    best_path = case_dir / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (case_dir / expected_best_target).resolve()


def test_checkpoint_freq_flag_controls_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=3, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=7)

    saved_epochs: list[int] = []
    val_losses = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.7, 5: 0.6, 6: 0.3}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == [0, 2, 5]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "5").resolve()


def test_save_and_load_best_val_loss(tmp_path: Path):
    cp = SingleGPUCheckpointer(str(tmp_path))

    # No file yet
    assert cp.load_best_val_loss() is None

    # Save and reload
    cp.save_best_val_loss(0.123456)
    assert cp.load_best_val_loss() == pytest.approx(0.123456)

    # Overwrite
    cp.save_best_val_loss(0.05)
    assert cp.load_best_val_loss() == pytest.approx(0.05)


def test_best_val_loss_restored_on_resume(tmp_path: Path):
    (tmp_path / "4").mkdir()

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.save_best_val_loss(0.42)

    trainer = Trainer.__new__(Trainer)
    trainer.resume_from_checkpoint = True
    trainer.is_distributed = False
    trainer.local_rank = 0
    trainer.checkpointer = cp

    trainer.current_epoch = cp.previous_epoch + 1
    trainer.global_step = 0
    trainer.best_val_loss = float("inf")

    # Simulate the load that setup_trainer does
    saved = trainer.checkpointer.load_best_val_loss()
    if saved is not None:
        trainer.best_val_loss = saved

    assert trainer.best_val_loss == pytest.approx(0.42)
