"""Run an external checkpoint evaluator without inheriting the training world."""

import json
import math
import os
import shlex
import signal
import stat
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

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
