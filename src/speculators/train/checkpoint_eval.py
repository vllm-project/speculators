"""Run an external checkpoint evaluator without inheriting the training world."""

import json
import logging
import math
import os
import shlex
import shutil
import signal
import stat
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("speculators")

_DISTRIBUTED_ENV = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
}
_MAX_METRICS_BYTES = 1 << 20
_SNAPSHOT_SKIP_FILES = frozenset({"optimizer_state_dict.pt", "scheduler_state_dict.pt"})


class _EvaluationCancelledError(Exception):
    """Internal signal used to stop a background evaluator promptly."""


@dataclass(frozen=True)
class CheckpointEvalResult:
    """Completed serving evaluation metrics ready for main-thread logging."""

    step: int
    epoch: int
    metrics: dict[str, float]
    output_dir: Path


@dataclass(frozen=True)
class _EvalJob:
    checkpoint: Path
    output_dir: Path
    step: int
    epoch: int
    cleanup_checkpoint: bool


def snapshot_checkpoint(source: Path, dest: Path) -> Path:
    """Copy a saved checkpoint's model files into *dest* without optimizer state.

    Training-only optimizer and scheduler blobs are omitted. Files are copied
    rather than hardlinked because subsequent sub-epoch saves overwrite
    ``config.json`` and weight files in place under the same epoch directory.
    """
    source = source.resolve()
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for root, dirs, files in os.walk(source):
            rel = Path(root).relative_to(source)
            target_dir = dest / rel
            target_dir.mkdir(parents=True, exist_ok=True)
            # Skip symlinked directories so a snapshot stays within the checkpoint.
            dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
            for name in files:
                if rel == Path() and name in _SNAPSHOT_SKIP_FILES:
                    continue
                src_file = Path(root) / name
                if src_file.is_symlink() or not src_file.is_file():
                    continue
                shutil.copy2(src_file, target_dir / name)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        if dest.parent.name == "_snapshots":
            with suppress(OSError):
                dest.parent.rmdir()
        raise
    return dest


def _wait_for_evaluator(
    process: subprocess.Popen,
    timeout: float,
    check_interrupted: Callable[[], None] | None,
) -> int:
    if check_interrupted is None:
        return process.wait(timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        check_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(0.25, remaining))
        except subprocess.TimeoutExpired:
            pass


def run_checkpoint_eval(
    command: str,
    checkpoint: Path,
    output_dir: Path,
    step: int,
    timeout: float,
    *,
    check_interrupted: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Run a POSIX evaluator command and read its finite scalar ``metrics.json``.

    The command receives ``SPECULATORS_EVAL_CHECKPOINT``, ``SPECULATORS_EVAL_STEP``
    and ``SPECULATORS_EVAL_OUTPUT_DIR``. It owns serving setup and evaluation GPU
    selection. A fresh output directory prevents stale metrics from a previous
    invocation being mistaken for a successful result.
    """
    argv = shlex.split(command)
    if not argv:
        raise ValueError("Checkpoint evaluation command must not be empty")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Checkpoint evaluation timeout must be positive and finite")
    checkpoint, output_dir = checkpoint.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "command": argv,
                "checkpoint": str(checkpoint),
                "step": step,
            },
            indent=2,
        )
        + "\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _DISTRIBUTED_ENV and not k.startswith("TORCHELASTIC_")
    }
    env.update(
        SPECULATORS_EVAL_CHECKPOINT=str(checkpoint),
        SPECULATORS_EVAL_STEP=str(step),
        SPECULATORS_EVAL_OUTPUT_DIR=str(output_dir),
    )
    with (output_dir / "evaluator.log").open("w") as log:
        with subprocess.Popen(  # noqa: S603
            argv,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            try:
                returncode = _wait_for_evaluator(process, timeout, check_interrupted)
            finally:
                # A serving engine can leave workers behind even after its parent
                # exits. Reap the whole session on success, failure or cancellation.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        if returncode:
            raise subprocess.CalledProcessError(returncode, argv)

    if check_interrupted is not None:
        check_interrupted()
    return _read_metrics(output_dir / "metrics.json")


def _read_metrics(path: Path) -> dict[str, float]:
    # A FIFO or device must not strand peers after the command's timeout. Open
    # without following symlinks and check the opened descriptor to avoid races.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Evaluation metrics must be a regular file")
        payload = handle.read(_MAX_METRICS_BYTES + 1)
    if len(payload) > _MAX_METRICS_BYTES:
        raise ValueError("Evaluation metrics exceed the 1 MiB limit")
    try:
        metrics = json.loads(payload)
    except RecursionError as exc:
        raise ValueError("Evaluation metrics JSON is too deeply nested") from exc
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Evaluation metrics must be a nonempty scalar mapping")
    result = {}
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Invalid evaluation metrics entry")
        try:
            scalar = float(value)
        except OverflowError as exc:
            raise ValueError("Evaluation metrics value is too large") from exc
        if not math.isfinite(scalar):
            raise ValueError("Non-finite evaluation metrics value")
        result[name] = scalar
    return result


class AsyncCheckpointEvaluator:
    """Single-worker background evaluator with a latest-wins pending slot.

    Keeps the dedicated evaluation GPU continuously utilized on the most recent
    checkpoint snapshot without pausing training or building an unbounded queue.
    Completed metrics are collected via :meth:`poll_results` or :meth:`drain`
    so the training main thread performs all metric-logger calls.
    """

    def __init__(self, command: str, timeout: float) -> None:
        self._command = command
        self._timeout = timeout
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: _EvalJob | None = None
        self._closed = False
        self._cancel = threading.Event()
        self._results: deque[CheckpointEvalResult] = deque()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _cleanup_job(job: _EvalJob) -> None:
        if job.cleanup_checkpoint:
            shutil.rmtree(job.checkpoint, ignore_errors=True)
            if job.checkpoint.parent.name == "_snapshots":
                with suppress(OSError):
                    job.checkpoint.parent.rmdir()

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise _EvaluationCancelledError

    def submit(
        self,
        checkpoint: Path,
        output_dir: Path,
        step: int,
        epoch: int,
        *,
        cleanup_checkpoint: bool = False,
    ) -> None:
        job = _EvalJob(
            checkpoint=checkpoint,
            output_dir=output_dir,
            step=step,
            epoch=epoch,
            cleanup_checkpoint=cleanup_checkpoint,
        )
        superseded: _EvalJob | None = None
        with self._cond:
            if self._closed:
                superseded = job
            else:
                if self._pending is not None:
                    logger.info(
                        "Superseding pending checkpoint evaluation at step %d "
                        "with step %d",
                        self._pending.step,
                        step,
                    )
                    superseded = self._pending
                self._pending = job
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run_loop,
                        name="speculators-serving-eval",
                        daemon=True,
                    )
                    self._thread.start()
                self._cond.notify()
        if superseded is not None:
            self._cleanup_job(superseded)

    def _run_loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._pending is None:
                    return
                job = self._pending
                self._pending = None
                if self._cancel.is_set():
                    self._cleanup_job(job)
                    return
            try:
                metrics = run_checkpoint_eval(
                    self._command,
                    job.checkpoint,
                    job.output_dir,
                    job.step,
                    self._timeout,
                    check_interrupted=self._check_cancelled,
                )
            except _EvaluationCancelledError:
                logger.info("Cancelled checkpoint evaluation at step %d", job.step)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Checkpoint evaluation failed at step %d; continuing training. "
                    "Artifacts: %s",
                    job.step,
                    job.output_dir,
                    exc_info=True,
                )
            else:
                with self._cond:
                    self._results.append(
                        CheckpointEvalResult(
                            step=job.step,
                            epoch=job.epoch,
                            metrics=metrics,
                            output_dir=job.output_dir,
                        )
                    )
            finally:
                self._cleanup_job(job)

    def poll_results(self) -> list[CheckpointEvalResult]:
        with self._lock:
            results = list(self._results)
            self._results.clear()
            return results

    def drain(self) -> list[CheckpointEvalResult]:
        """Finish any running and pending evaluation and return unread results."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            try:
                while thread.is_alive():
                    thread.join(timeout=0.25)
            except BaseException:
                self.cancel()
                raise
        return self.poll_results()

    def cancel(self) -> None:
        """Abort the active evaluator, drop pending work, and clean snapshots."""
        with self._cond:
            self._closed = True
            self._cancel.set()
            pending = self._pending
            self._pending = None
            self._cond.notify_all()
            thread = self._thread
        if pending is not None:
            self._cleanup_job(pending)
        if thread is not None:
            thread.join(timeout=5.0)
