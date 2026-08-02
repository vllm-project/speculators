"""Unit tests: run_training must stop at max_steps and persist the final state."""

from types import SimpleNamespace

from speculators.train.trainer import Trainer


class _LoopProbe(Trainer):
    """Trainer with the training internals stubbed out to probe the epoch loop."""

    def __init__(self, num_epochs: int, max_steps: int | None, epoch_len: int):
        self.config = SimpleNamespace(num_epochs=num_epochs, max_steps=max_steps)
        self.global_step = 0
        self.current_epoch = 0
        self.is_distributed = False
        self.val_loader = None
        self.epoch_len = epoch_len
        self.epochs_entered: list[int] = []
        self.save_calls: list[tuple[int, bool]] = []

    def train_epoch(self, epoch: int) -> None:
        self.epochs_entered.append(epoch)
        # Mirror the real inner-loop behavior: step until the epoch ends or
        # max_steps is hit.
        for _ in range(self.epoch_len):
            self.global_step += 1
            if self._reached_max_steps():
                break

    def maybe_save_checkpoint(self, epoch, local_step=0, force=False) -> None:
        self.save_calls.append((epoch, force))

    def maybe_update_best(self, epoch, val_metrics) -> None:
        pass

    def val_epoch(self, epoch):
        return None


def test_epoch_loop_stops_once_max_steps_reached():
    trainer = _LoopProbe(num_epochs=10, max_steps=15, epoch_len=10)
    trainer.run_training()
    # max_steps lands mid-epoch 1; no further epoch may run an optimizer step.
    assert trainer.epochs_entered == [0, 1]
    assert trainer.global_step == 15


def test_final_segment_checkpoint_is_forced():
    # The final segment before max_steps must be saved even when the
    # checkpoint_freq / save_best gates would skip that epoch.
    trainer = _LoopProbe(num_epochs=10, max_steps=15, epoch_len=10)
    trainer.run_training()
    assert trainer.save_calls == [(0, False), (1, True)]


def test_epoch_loop_unaffected_without_max_steps():
    trainer = _LoopProbe(num_epochs=3, max_steps=None, epoch_len=4)
    trainer.run_training()
    assert trainer.epochs_entered == [0, 1, 2]
    assert trainer.global_step == 12
    assert all(force is False for _, force in trainer.save_calls)
