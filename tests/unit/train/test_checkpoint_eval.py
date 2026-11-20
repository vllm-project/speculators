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
from speculators.train.checkpoint_eval import run_checkpoint_eval


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
