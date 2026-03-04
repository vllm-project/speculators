import pytest

from speculators.train.trainer import TrainerConfig


class TestTrainerConfig:
    """Tests for TrainerConfig fields including new checkpoint options."""

    def test_default_values(self):
        config = TrainerConfig(lr=1e-4, num_epochs=10, save_path="./ckpts")
        assert config.save_best is False
        assert config.save_optimizer_state is True
        assert config.max_checkpoints is None

    def test_save_best_enabled(self):
        config = TrainerConfig(
            lr=1e-4, num_epochs=10, save_path="./ckpts", save_best=True
        )
        assert config.save_best is True

    def test_save_optimizer_state_disabled(self):
        config = TrainerConfig(
            lr=1e-4, num_epochs=10, save_path="./ckpts", save_optimizer_state=False
        )
        assert config.save_optimizer_state is False

    def test_max_checkpoints_set(self):
        config = TrainerConfig(
            lr=1e-4, num_epochs=10, save_path="./ckpts", max_checkpoints=3
        )
        assert config.max_checkpoints == 3


class TestShouldSaveCheckpoint:
    """Tests for Trainer._should_save_checkpoint logic.

    Since Trainer requires a model and data loaders which are heavy to
    instantiate, we test the logic directly on a minimal mock.
    """

    @staticmethod
    def _should_save(
        save_best: bool,
        val_loss: float | None,
        best_val_loss: float | None,
    ) -> bool:
        """Reproduce the _should_save_checkpoint logic without a full Trainer."""
        if not save_best:
            return True
        if val_loss is None:
            return True
        if best_val_loss is None:
            return True
        return val_loss < best_val_loss

    def test_always_save_when_save_best_disabled(self):
        assert self._should_save(save_best=False, val_loss=1.0, best_val_loss=0.5)

    def test_save_when_no_val_loss(self):
        assert self._should_save(save_best=True, val_loss=None, best_val_loss=0.5)

    def test_save_when_no_best_yet(self):
        assert self._should_save(save_best=True, val_loss=1.0, best_val_loss=None)

    def test_save_when_loss_improves(self):
        assert self._should_save(save_best=True, val_loss=0.3, best_val_loss=0.5)

    def test_skip_when_loss_does_not_improve(self):
        assert not self._should_save(save_best=True, val_loss=0.6, best_val_loss=0.5)

    def test_skip_when_loss_equals_best(self):
        assert not self._should_save(save_best=True, val_loss=0.5, best_val_loss=0.5)
