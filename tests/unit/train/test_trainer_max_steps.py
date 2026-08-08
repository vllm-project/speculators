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
        self.save_calls: list[tuple[int, int, bool]] = []

    def train_epoch(self, epoch: int) -> int:
        self.epochs_entered.append(epoch)
        # Mirror the real inner-loop behavior: step until the epoch ends or
        # max_steps is hit, returning the partial-epoch cursor in the latter case.
        for local_step in range(1, self.epoch_len + 1):
            self.global_step += 1
            if self._reached_max_steps():
                return local_step if local_step < self.epoch_len else 0
        return 0

    def maybe_save_checkpoint(self, epoch, local_step=0, force=False) -> None:
        self.save_calls.append((epoch, local_step, force))

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
    assert trainer.save_calls == [(0, 0, False), (1, 5, True)]


def test_forced_checkpoint_records_the_partial_epoch_cursor():
    # max_steps=15 with 10 steps per epoch stops five batches into epoch 1.
    # Recording local_step=0 there would mark the epoch complete, so a resume
    # with a higher max_steps would silently skip the five untrained batches.
    trainer = _LoopProbe(num_epochs=10, max_steps=15, epoch_len=10)
    trainer.run_training()
    epoch, local_step, force = trainer.save_calls[-1]
    assert (epoch, local_step, force) == (1, 5, True)


def test_max_steps_on_the_last_batch_records_end_of_epoch():
    # The epoch did complete, so the checkpoint must stay an end-of-epoch one.
    trainer = _LoopProbe(num_epochs=10, max_steps=20, epoch_len=10)
    trainer.run_training()
    assert trainer.save_calls[-1] == (1, 0, True)


def test_epoch_loop_unaffected_without_max_steps():
    trainer = _LoopProbe(num_epochs=3, max_steps=None, epoch_len=4)
    trainer.run_training()
    assert trainer.epochs_entered == [0, 1, 2]
    assert trainer.global_step == 12
    assert all(force is False for _, _, force in trainer.save_calls)
