"""Graceful shutdown handler for saving checkpoints on interrupt.

When Ctrl+C is pressed during training:
- torchrun intercepts SIGINT and sends SIGINT/SIGTERM to all worker processes
- In single-GPU mode, the process receives SIGINT directly

This module handles both cases by registering handlers for SIGINT and SIGTERM.
The first signal raises a TrainingInterruptedError exception that unwinds the call
stack -- this works even when a process is stuck in a NCCL collective, data
loading, or a GPU kernel. The exception is caught at the run_training level
to attempt a checkpoint save.

Key design decisions:
- The handler only raises in the process that called install() (tracked via
  PID). Forked dataloader workers inherit the handler but ignore the signal,
  preventing worker crashes.
- After the first signal, subsequent rapid re-sends (torchrun sends SIGINT
  to the process group AND then again directly to each child) are silently
  ignored. Default handlers are restored explicitly before the save attempt,
  so a deliberate second Ctrl+C during the save will force immediate exit.
"""

import logging
import os
import signal
import threading
from functools import wraps
from typing import Any

logger = logging.getLogger("speculators")

# Default timeout for coordinated shutdown save (seconds)
DEFAULT_SHUTDOWN_TIMEOUT = 120


class TrainingInterruptedError(Exception):
    """Raised by the signal handler to interrupt training for checkpoint save."""


class GracefulShutdownHandler:
    """Manages graceful shutdown with checkpoint saving on interrupt.

    First interrupt: raises TrainingInterruptedError to break out of stuck code.
    Subsequent rapid signals (from torchrun re-sends): silently ignored.
    After restore(): default handlers are active, so another Ctrl+C kills.
    """

    def __init__(self, timeout: int = DEFAULT_SHUTDOWN_TIMEOUT):
        self._interrupted = False
        self._lock = threading.Lock()
        self._original_sigint: Any = None
        self._original_sigterm: Any = None
        self._timeout = timeout
        self._owner_pid: int | None = None

    def install(self):
        """Register signal handlers for SIGINT and SIGTERM."""
        self._owner_pid = os.getpid()
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def restore(self):
        """Restore original signal handlers.

        Called before the checkpoint save attempt so that a deliberate
        second Ctrl+C during the save causes immediate exit.
        """
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handler(self, signum, frame):  # noqa: ARG002
        # Only handle in the process that installed the handler.
        # Forked dataloader workers inherit signal handlers but should
        # not raise TrainingInterruptedError (it would crash the worker).
        if os.getpid() != self._owner_pid:
            return

        sig_name = signal.Signals(signum).name
        with self._lock:
            if self._interrupted:
                # Already interrupted. Ignore rapid re-sends from torchrun.
                # Default handlers will be restored via restore() before the
                # save attempt, so a deliberate second Ctrl+C will force exit.
                return
            self._interrupted = True
            logger.warning(
                f"Received {sig_name} — interrupting training to save checkpoint. "
                "Send again to force immediate exit."
            )
            raise TrainingInterruptedError(sig_name)

    @property
    def timeout(self) -> int:
        return self._timeout


def with_graceful_shutdown(
    save_label: str = "interrupted",
    timeout: int = DEFAULT_SHUTDOWN_TIMEOUT,
):
    """Decorator that wraps a Trainer method with graceful shutdown handling.

    The decorated method's `self` must have `maybe_save_checkpoint(label)` and
    `checkpointer.path`.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            handler = GracefulShutdownHandler(timeout=timeout)
            handler.install()

            try:
                return fn(self, *args, **kwargs)
            except TrainingInterruptedError:
                handler.restore()

                logger.warning(
                    "Training interrupted — attempting to save checkpoint "
                    f"(timeout={handler.timeout}s, send Ctrl+C again to force exit)..."
                )

                def _watchdog():
                    logger.error(
                        f"Interrupt checkpoint save timed out after {handler.timeout}s "
                        "— forcing exit"
                    )
                    os._exit(1)

                timer = threading.Timer(handler.timeout, _watchdog)
                timer.daemon = True
                timer.start()

                try:
                    self.maybe_save_checkpoint(save_label)
                    logger.info(
                        "Interrupt checkpoint saved to "
                        f"'{self.checkpointer.path / save_label}'"
                    )
                except Exception:
                    logger.exception("Failed to save interrupt checkpoint")
                finally:
                    timer.cancel()

        return wrapper

    return decorator
