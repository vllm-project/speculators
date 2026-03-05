from pathlib import Path
from types import SimpleNamespace

import pytest

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer


def _make_minimal_trainer(tmp_path: Path, checkpoint_freq: int, save_best: bool):
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
        num_epochs=0,
        save_path=str(tmp_path),
    )
    trainer.best_val_loss = float("inf")
    trainer.best_epoch = None
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.is_distributed = False
    trainer.val_loader = object()
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))
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

def test_run_training_updates_checkpoint_best_among_saved_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=2, save_best=True)
    trainer.config.num_epochs = 4

    saved_epochs = []
    val_losses = {
        0: 0.9,
        1: 0.6,  # saved
        2: 0.1,  # not saved, should not become checkpoint_best
        3: 0.7,  # saved and better than epoch 1 among saved epochs
    }

    def fake_train_epoch(epoch: int):
        return None

    def fake_val_epoch(epoch: int):
        return {"loss_epoch": val_losses[epoch]}

    def fake_save_checkpoint(epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.train_epoch = fake_train_epoch
    trainer.val_epoch = fake_val_epoch
    trainer.save_checkpoint = fake_save_checkpoint

    trainer.run_training()

    assert saved_epochs == [0, 1, 3]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()

    # Best among saved checkpoints is epoch 1 (loss 0.6), not epoch 2 (loss 0.1)
    # because epoch 2 was not saved under checkpoint_freq=2.
    assert best_path.resolve() == (tmp_path / "1").resolve()
    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == 0.6
