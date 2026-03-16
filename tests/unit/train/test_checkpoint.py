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
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.is_distributed = False
    trainer.val_loader = object()
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))

    trainer.model = object()
    trainer.opt = object()
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
    trainer.config.num_epochs = 4

    saved_epochs = []
    val_losses = {
        0: 0.9,
        1: 0.6,  # saved
        2: 0.1,  # not saved, should not become checkpoint_best
        3: 0.7,  # saved and worse than epoch 1 among saved epochs
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
    trainer.checkpointer.save_checkpoint = fake_cp_save_checkpoint
    trainer.checkpointer.save_scheduler_state_dict = lambda *_args, **_kwargs: None

    trainer.run_training()

    assert saved_epochs == [0, 1, 3]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()

    # Best among saved checkpoints is epoch 1 (loss 0.6), not epoch 2 (loss 0.1)
    # because epoch 2 was not saved under checkpoint_freq=2.
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
        # save_best=False: respect checkpoint_freq, keep all saved dirs.
        (False, 3, [0, 2], {"0", "2"}, "2"),
        # save_best=True: ignore checkpoint_freq,
        # save only when new best occurs, keep only best dir.
        (True, 3, [0, 1, 3], {"3"}, "3"),
    ],
)

def test_save_best_flag_changes_checkpoint_behavior(
    tmp_path: Path,
    save_best: bool,
    checkpoint_freq: int,
    expected_saved: list[int],
    expected_remaining_dirs: set[str],
    expected_best_target: str,
):
    case_dir = tmp_path / ("save_best" if save_best else "save_freq")
    case_dir.mkdir()
    trainer = _make_minimal_trainer(case_dir,
                                    checkpoint_freq=checkpoint_freq,
                                    save_best=save_best
                                    )
    trainer.config.num_epochs = 4

    saved_epochs: list[int] = []
    # Designed so that epoch 2 is *not* a new best, but epoch 3 is.
    val_losses = {0: 0.9, 1: 0.8, 2: 0.85, 3: 0.7}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (case_dir / str(epoch)).mkdir(exist_ok=True)

    trainer.checkpointer.save_checkpoint = fake_cp_save_checkpoint
    trainer.checkpointer.save_scheduler_state_dict = lambda *_args, **_kwargs: None

    trainer.run_training()

    assert saved_epochs == expected_saved

    remaining_dirs = {p.name
                      for p in case_dir.iterdir()
                      if p.is_dir() and p.name.isdigit()
                      }
    assert remaining_dirs == expected_remaining_dirs

    best_path = case_dir / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (case_dir / expected_best_target).resolve()

def test_checkpoint_freq_flag_controls_saves(tmp_path: Path):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=3, save_best=False)
    trainer.config.num_epochs = 7

    saved_epochs: list[int] = []
    # Epoch 1 has the best loss, but it should NOT be saved when checkpoint_freq=3.
    val_losses = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.7, 5: 0.6, 6: 0.3}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.checkpointer.save_checkpoint = fake_cp_save_checkpoint
    trainer.checkpointer.save_scheduler_state_dict = lambda *_args, **_kwargs: None

    trainer.run_training()

    # With checkpoint_freq=3, Trainer saves epoch 0 and then every 3 epochs: 2, 5, ...
    assert saved_epochs == [0, 2, 5]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    # Best among saved (0,2,5) is epoch 5 (loss 0.6)
    assert best_path.resolve() == (tmp_path / "5").resolve()


def test_init_best_val_loss_on_resume_with_and_without_checkpoint_best(tmp_path: Path):
    # Case A: checkpoint_best exists and points to a valid epoch.
    (tmp_path / "0").mkdir()
    (tmp_path / "4").mkdir()
    (tmp_path / "checkpoint_best").symlink_to("0", target_is_directory=True)

    trainer = Trainer.__new__(Trainer)
    trainer.is_distributed = False
    trainer.local_rank = 0
    trainer.val_loader = object()
    trainer.best_val_loss = float("inf")
    trainer.model = object()
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))

    calls: list[int] = []

    def fake_load_for_epoch(_model, epoch: int, float_dtype=None):
        calls.append(epoch)

    trainer.checkpointer.load_model_state_dict_for_epoch = fake_load_for_epoch
    trainer.val_epoch = lambda epoch: {"loss_epoch": 0.123}

    trainer.init_best_val_loss_from_checkpoint_best()

    assert trainer.best_val_loss == 0.123
    # Should load best epoch then restore last epoch (previous_epoch=4)
    assert calls == [0, 4]

    # Case B: checkpoint_best missing/broken -> leave best_val_loss as inf.
    tmp2 = tmp_path / "no_best"
    tmp2.mkdir()
    (tmp2 / "4").mkdir()

    trainer2 = Trainer.__new__(Trainer)
    trainer2.is_distributed = False
    trainer2.local_rank = 0
    trainer2.val_loader = object()
    trainer2.best_val_loss = float("inf")
    trainer2.model = object()
    trainer2.checkpointer = SingleGPUCheckpointer(str(tmp2))

    calls2: list[int] = []
    trainer2.checkpointer.load_model_state_dict_for_epoch = (
        lambda *_args, **_kwargs: calls2.append(1)
    )
    trainer2.val_epoch = lambda epoch: {"loss_epoch": 0.001}

    trainer2.init_best_val_loss_from_checkpoint_best()

    assert trainer2.best_val_loss == float("inf")
    assert calls2 == []
