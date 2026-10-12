"""The graceful-shutdown decorator must not leave its handlers installed.

`with_graceful_shutdown()` installs SIGINT/SIGTERM handlers for the duration of
one training call. It used to restore them only when it caught
`TrainingInterruptedError`, so a normal return or an unrelated exception left
this call's handlers in place. In a longer-lived process a later interrupt then
entered a handler belonging to a training run that had already finished.
"""

import signal
from pathlib import Path

import pytest

from speculators.train.graceful_shutdown import (
    GracefulShutdownHandler,
    TrainingInterruptedError,
    with_graceful_shutdown,
)


@pytest.fixture
def original_handlers():
    """Snapshot the process handlers and put them back afterwards."""
    signals = (signal.SIGINT, signal.SIGTERM)
    before = {sig: signal.getsignal(sig) for sig in signals}
    yield before
    for sig, handler in before.items():
        signal.signal(sig, handler)


def _installed(before):
    """True when the process handlers are the ones from before the call."""
    return all(signal.getsignal(sig) == handler for sig, handler in before.items())


class _Checkpointer:
    path = Path()


class _Trainer:
    def __init__(self):
        self.checkpointer = _Checkpointer()
        self.saved = []

    def maybe_save_checkpoint(self, label):
        self.saved.append(label)


def test_handlers_are_restored_after_a_normal_return(original_handlers):
    @with_graceful_shutdown()
    def train(self):
        # The handlers are this call's while it runs; that part already worked.
        assert not _installed(original_handlers)
        return "finished"

    assert train(_Trainer()) == "finished"
    assert _installed(original_handlers), "handlers outlived a successful call"


def test_handlers_are_restored_after_an_unrelated_exception(original_handlers):
    @with_graceful_shutdown()
    def train(self):
        raise ValueError("unrelated")

    with pytest.raises(ValueError, match="unrelated"):
        train(_Trainer())
    assert _installed(original_handlers), "handlers outlived a failed call"


def test_handlers_are_restored_after_an_interruption(original_handlers):
    """The path that already restored must keep doing so."""

    @with_graceful_shutdown()
    def train(self):
        raise TrainingInterruptedError("SIGINT")

    trainer = _Trainer()
    train(trainer)
    assert trainer.saved == ["interrupted"], "the checkpoint save no longer runs"
    assert _installed(original_handlers)


def test_handlers_are_restored_when_the_checkpoint_save_itself_fails(original_handlers):
    """A save failure is caught and logged, and must not strand the handlers."""

    class Failing(_Trainer):
        def maybe_save_checkpoint(self, label):
            raise RuntimeError("disk full")

    @with_graceful_shutdown()
    def train(self):
        raise TrainingInterruptedError("SIGTERM")

    train(Failing())
    assert _installed(original_handlers)


def test_nested_calls_unwind_to_the_original_handlers(original_handlers):
    """Repeated and nested decorated calls must not capture stale handlers."""

    @with_graceful_shutdown()
    def inner(self):
        return "inner"

    @with_graceful_shutdown()
    def outer(self):
        outer_handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        assert inner(self) == "inner"
        # The inner call restores what it found, which is the outer call's.
        for sig, handler in outer_handlers.items():
            assert signal.getsignal(sig) == handler
        return "outer"

    assert outer(_Trainer()) == "outer"
    assert _installed(original_handlers)

    # And running the same decorated function twice ends where it started.
    assert inner(_Trainer()) == "inner"
    assert _installed(original_handlers)


def test_restore_is_idempotent(original_handlers):
    """The decorator restores in both the except branch and the finally block,
    so a second restore must not undo a handler installed in between."""
    handler = GracefulShutdownHandler()
    handler.install()
    handler.restore()
    assert _installed(original_handlers)

    def someone_elses_handler(signum, frame):
        raise AssertionError("never called")

    signal.signal(signal.SIGINT, someone_elses_handler)
    handler.restore()
    assert signal.getsignal(signal.SIGINT) is someone_elses_handler


def test_install_is_idempotent(original_handlers):
    """A second install() must not capture this handler as the original."""
    handler = GracefulShutdownHandler()
    handler.install()
    handler.install()
    handler.restore()
    assert _installed(original_handlers)
