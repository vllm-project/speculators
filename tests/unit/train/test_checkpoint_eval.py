"""Exercise the evaluator protocol with real subprocesses, without serving a model."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from speculators.train import checkpoint_eval
from speculators.train.checkpoint_eval import (
    AsyncCheckpointEvaluator,
    run_checkpoint_eval,
    snapshot_checkpoint,
)


def _command(tmp_path: Path, body: str) -> str:
    script = tmp_path / "eval script.py"
    script.write_text(body)
    return f'{sys.executable} "{script}"'


def test_runs_with_exact_checkpoint_and_isolated_training_environment(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.setenv(key, "training-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    command = _command(
        tmp_path,
        """
import json, os
from pathlib import Path
assert Path(os.environ["SPECULATORS_EVAL_CHECKPOINT"]).joinpath("config.json").exists()
assert os.environ["SPECULATORS_EVAL_STEP"] == "17"
assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
assert not any(k in os.environ for k in (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
))
out = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
out.joinpath("metrics.json").write_text(json.dumps({
    "mean_acceptance_length": 3.25, "acceptance_rate_pos_1": 0.875,
}))
print("evaluated checkpoint")
""",
    )
    output = tmp_path / "output"

    metrics = run_checkpoint_eval(command, checkpoint, output, step=17, timeout=10)

    assert metrics == {"mean_acceptance_length": 3.25, "acceptance_rate_pos_1": 0.875}
    assert "evaluated checkpoint" in (output / "evaluator.log").read_text()
    manifest = json.loads((output / "evaluation.json").read_text())
    assert manifest["checkpoint"] == str(checkpoint.resolve())
    assert manifest["step"] == 17


@pytest.mark.parametrize(
    "payload",
    ["{}", "[]", '{"mal": true}', '{"mal": "3"}', '{"mal": NaN}', '{"mal": Infinity}'],
)
def test_rejects_invalid_metric_results(tmp_path, payload):
    command = _command(
        tmp_path,
        f"""
import os
from pathlib import Path
Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"]).joinpath("metrics.json").write_text({payload!r})
""",
    )
    with pytest.raises(ValueError, match="metrics"):
        run_checkpoint_eval(command, tmp_path, tmp_path / "out", step=1, timeout=10)


def test_missing_result_and_failed_process_are_not_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_checkpoint_eval(
            _command(tmp_path, "print('no result')"),
            tmp_path,
            tmp_path / "missing",
            step=1,
            timeout=10,
        )
    with pytest.raises(subprocess.CalledProcessError):
        run_checkpoint_eval(
            _command(tmp_path, "raise SystemExit(7)"),
            tmp_path,
            tmp_path / "failed",
            step=1,
            timeout=10,
        )


@pytest.mark.parametrize(
    "payload",
    ["[" * 100_000 + "]" * 100_000, json.dumps({"x" * (2 << 20): 1})],
    ids=["nested", "oversized"],
)
def test_rejects_deeply_nested_or_oversized_results(tmp_path, payload):
    source = tmp_path / "payload.json"
    source.write_text(payload)
    command = _command(
        tmp_path,
        "import os, shutil\n"
        "from pathlib import Path\n"
        f"shutil.copyfile({str(source)!r}, "
        "Path(os.environ['SPECULATORS_EVAL_OUTPUT_DIR']) / 'metrics.json')\n",
    )
    with pytest.raises(ValueError, match="metrics"):
        run_checkpoint_eval(command, tmp_path, tmp_path / "out", step=1, timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="FIFOs require POSIX")
def test_rejects_fifo_without_blocking_beyond_evaluator_timeout(tmp_path):
    command = _command(
        tmp_path,
        "import os\nfrom pathlib import Path\n"
        "os.mkfifo(Path(os.environ['SPECULATORS_EVAL_OUTPUT_DIR']) / 'metrics.json')\n",
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import runpy\nfrom pathlib import Path\n"
        f"run = runpy.run_path({checkpoint_eval.__file__!r})['run_checkpoint_eval']\n"
        "try:\n"
        f"    run({command!r}, Path({str(tmp_path)!r}), "
        f"Path({str(tmp_path / 'out')!r}), step=1, timeout=1)\n"
        "except ValueError:\n    pass\n"
        "else:\n    raise AssertionError('FIFO accepted as metrics')\n",
    )
    subprocess.run([sys.executable, str(driver)], timeout=3, check=True)  # noqa: S603


def test_does_not_reuse_stale_results(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "metrics.json").write_text('{"mean_acceptance_length": 8}')
    with pytest.raises(FileExistsError):
        run_checkpoint_eval(
            _command(tmp_path, "pass"), tmp_path, output, step=2, timeout=10
        )


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
@pytest.mark.parametrize("timeout", [0.3, 10])
def test_cleans_descendant_processes_on_timeout_and_success(tmp_path, timeout):
    child_pid = tmp_path / "child.pid"
    command = _command(
        tmp_path,
        f"""
import json, os, subprocess, sys, time
from pathlib import Path
code = "import signal,time; "
code += "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
child = subprocess.Popen([sys.executable, "-c", code])
Path({str(child_pid)!r}).write_text(str(child.pid))
if {timeout} < 1:
    time.sleep(60)
out = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
out.joinpath("metrics.json").write_text('{{"mean_acceptance_length": 2}}')
""",
    )
    if timeout < 1:
        with pytest.raises(subprocess.TimeoutExpired):
            run_checkpoint_eval(
                command, tmp_path, tmp_path / "out", step=1, timeout=timeout
            )
    else:
        run_checkpoint_eval(
            command, tmp_path, tmp_path / "out", step=1, timeout=timeout
        )
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().split()[2] == "Z":
                return
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        pytest.fail(f"evaluator leaked worker {pid}")
    finally:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def test_snapshot_checkpoint_copies_weights_and_skips_optimizer_state(tmp_path):
    source = tmp_path / "epoch0"
    (source / "tokenizer").mkdir(parents=True)
    weights = source / "model.safetensors"
    weights.write_bytes(b"draft-weights-v1")
    (source / "config.json").write_text('{"step": 1}')
    (source / "tokenizer" / "vocab.txt").write_text("tok")
    (source / "optimizer_state_dict.pt").write_bytes(b"huge-optimizer")
    (source / "scheduler_state_dict.pt").write_bytes(b"scheduler")

    snap = snapshot_checkpoint(source, tmp_path / "snapshot")

    assert (snap / "model.safetensors").stat().st_ino != weights.stat().st_ino
    assert (snap / "tokenizer" / "vocab.txt").read_text() == "tok"
    assert not (snap / "optimizer_state_dict.pt").exists()
    assert not (snap / "scheduler_state_dict.pt").exists()

    # In-place overwrites or removals of the source must not affect the snapshot.
    weights.unlink()
    (source / "config.json").write_text('{"step": 2}')
    assert (snap / "model.safetensors").read_bytes() == b"draft-weights-v1"
    assert (snap / "config.json").read_text() == '{"step": 1}'


def test_async_evaluator_latest_wins_supersedes_pending_and_cleans_snapshots(tmp_path):
    step1_started = tmp_path / "step1.started"
    release_step1 = tmp_path / "step1.release"
    command = _command(
        tmp_path,
        f"""
import json, os, time
from pathlib import Path
step = int(os.environ["SPECULATORS_EVAL_STEP"])
ckpt = Path(os.environ["SPECULATORS_EVAL_CHECKPOINT"])
tag = json.loads((ckpt / "config.json").read_text())["step"]
assert tag == step
if step == 1:
    Path({str(step1_started)!r}).touch()
    deadline = time.monotonic() + 10
    while not Path({str(release_step1)!r}).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
out = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
(out / "metrics.json").write_text(json.dumps({{"mal": float(step)}}))
""",
    )
    snaps = {}
    for step in (1, 2, 3):
        snap = tmp_path / "snaps" / f"step_{step}"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"step": step}))
        snaps[step] = snap

    evaluator = AsyncCheckpointEvaluator(command, timeout=15)
    evaluator.submit(
        snaps[1], tmp_path / "out1", step=1, epoch=0, cleanup_checkpoint=True
    )
    deadline = time.monotonic() + 10
    while not step1_started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert step1_started.exists()

    # Submit step 2 then step 3 while step 1 is active: step 2 is superseded
    # and its temporary snapshot is removed immediately.
    evaluator.submit(
        snaps[2], tmp_path / "out2", step=2, epoch=0, cleanup_checkpoint=True
    )
    evaluator.submit(
        snaps[3], tmp_path / "out3", step=3, epoch=0, cleanup_checkpoint=True
    )
    assert not snaps[2].exists()
    assert snaps[3].exists()

    release_step1.touch()
    results = evaluator.drain()

    assert [(r.step, r.metrics) for r in results] == [
        (1, {"mal": 1.0}),
        (3, {"mal": 3.0}),
    ]
    assert not snaps[1].exists()
    assert not snaps[3].exists()
    assert (tmp_path / "out1" / "metrics.json").exists()
    assert not (tmp_path / "out2").exists()
    assert (tmp_path / "out3" / "metrics.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_async_evaluator_cancel_kills_running_process_group_and_drops_pending(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    command = _command(
        tmp_path,
        f"""
import os, subprocess, sys, time
from pathlib import Path
code = (
    "import signal, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
)
child = subprocess.Popen([sys.executable, "-c", code])
Path({str(child_pid_file)!r}).write_text(str(child.pid))
time.sleep(60)
""",
    )
    snap1 = tmp_path / "snap1"
    snap2 = tmp_path / "snap2"
    snap1.mkdir()
    snap2.mkdir()

    evaluator = AsyncCheckpointEvaluator(command, timeout=60)
    evaluator.submit(snap1, tmp_path / "out1", step=1, epoch=0, cleanup_checkpoint=True)
    deadline = time.monotonic() + 10
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid_file.exists()
    evaluator.submit(snap2, tmp_path / "out2", step=2, epoch=0, cleanup_checkpoint=True)

    evaluator.cancel()

    assert not snap1.exists()
    assert not snap2.exists()
    pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().split()[2] == "Z":
                return
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        pytest.fail(f"async evaluator leaked worker {pid}")
    finally:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
