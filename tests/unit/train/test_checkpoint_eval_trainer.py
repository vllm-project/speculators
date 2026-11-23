"""Checkpoint-boundary tests without loading or training a language model."""

import importlib
import json
import logging
import os
import signal
import struct
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import speculators.train.checkpoint_eval as checkpoint_eval_mod
import speculators.train.checkpointer as checkpointer_mod
import speculators.train.distributed as distributed_mod
from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.config import TrainConfig
from speculators.train.graceful_shutdown import (
    TrainingInterruptedError,
    with_graceful_shutdown,
)
from speculators.train.logger import WandbHandler
from speculators.train.trainer import Trainer, TrainerConfig


def _trainer(
    tmp_path: Path,
    command: str | None = None,
    *,
    save_best=False,
    checkpoint_eval_mode="async",
    checkpoint_eval_steps=None,
) -> Trainer:
    trainer = cast("Any", object.__new__(Trainer))
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=1,
        save_path=str(tmp_path),
        checkpoint_eval_command=command,
        checkpoint_eval_timeout=10,
        checkpoint_eval_steps=checkpoint_eval_steps,
        checkpoint_eval_mode=checkpoint_eval_mode,
        save_best=save_best,
    )
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.current_epoch = 0
    trainer.train_epoch = lambda _epoch: None
    trainer.val_loader = []
    trainer.val_epoch = lambda _epoch: {"loss_epoch": 0.5}
    trainer.global_step = 17
    trainer.best_val_loss = float("inf")
    trainer.schedulers = []
    trainer.optimizers = []
    trainer.model = None
    trainer._last_checkpoint_eval = None
    trainer._checkpoint_evaluator = None

    def save(_model, _optimizers, epoch):
        path = tmp_path / str(epoch)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")

    def save_model(_model, output_dir, float_dtype=torch.bfloat16):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps({"step": trainer.global_step})
        )
        (output_dir / "model.safetensors").write_bytes(b"weights")

    trainer.checkpointer = SingleGPUCheckpointer(tmp_path)
    trainer.checkpointer.save_checkpoint = save
    trainer.checkpointer.save_model = save_model
    return cast("Trainer", trainer)


@pytest.fixture
def evaluator(tmp_path):
    script = tmp_path / "evaluate.py"
    script.write_text("""
import json, os
from pathlib import Path
checkpoint = Path(os.environ["SPECULATORS_EVAL_CHECKPOINT"])
assert checkpoint.joinpath("config.json").exists()
state = json.loads(checkpoint.joinpath("training_state.json").read_text())
assert state["global_step"] == 17
out = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
out.joinpath("metrics.json").write_text('{"mean_acceptance_length": 3.25}')
""")
    return f'{sys.executable} "{script}"'


def test_evaluates_complete_checkpoint_once_and_logs_its_training_step(
    tmp_path, evaluator, caplog
):
    trainer = _trainer(tmp_path, evaluator)
    with caplog.at_level(logging.INFO, logger="speculators.metrics"):
        trainer.run_training()
        trainer._evaluate_checkpoint(0)
    results = list((tmp_path / "serving_eval").glob("*/metrics.json"))
    assert len(results) == 1
    records = [
        r for r in caplog.records if isinstance(r.msg, dict) and "serving_eval" in r.msg
    ]
    assert len(records) == 1
    assert records[0].msg["serving_eval"] == {"mean_acceptance_length": 3.25}
    assert records[0].step == 17


def test_best_retention_preserves_metrics_and_failed_evaluation_logs(
    tmp_path, evaluator
):
    root = tmp_path / "checkpoints"
    trainer = _trainer(root, evaluator, save_best=True)
    historical = root / "serving_eval" / "old"
    historical.mkdir(parents=True)
    (historical / "metrics.json").write_text('{"acceptance_length": 2}')
    failed = root / "serving_eval" / "failed"
    failed.mkdir()
    (failed / "evaluator.log").write_text("previous failure")

    trainer.run_training()

    assert (historical / "metrics.json").exists()
    assert (failed / "evaluator.log").read_text() == "previous failure"
    assert len(list((root / "serving_eval").glob("*/metrics.json"))) == 2


@pytest.mark.parametrize("save_best", [False, True], ids=["periodic", "best-only"])
def test_resume_after_interrupted_epoch_evaluation(tmp_path, monkeypatch, save_best):
    root = tmp_path / "checkpoints"
    script = tmp_path / "interrupt_eval.py"
    script.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "if Path(os.environ['SPECULATORS_EVAL_CHECKPOINT']).name == '1':\n"
        "    os.kill(os.getppid(), signal.SIGTERM)\n"
        "    time.sleep(30)\n"
        "out = Path(os.environ['SPECULATORS_EVAL_OUTPUT_DIR'])\n"
        "out.joinpath('metrics.json').write_text('{\"mal\": 2}')\n"
    )
    trained, validated = [], []

    def run(resume):
        trainer = _trainer(
            root,
            f'"{sys.executable}" "{script}"',
            save_best=save_best,
            checkpoint_eval_mode="sync",
        )
        trainer.config = trainer.config._replace(num_epochs=3)
        trainer.resume_from_checkpoint = resume
        trainer.setup_trainer()

        def train_epoch(epoch):
            trained.append(epoch)
            trainer.global_step += 1

        def val_epoch(epoch):
            validated.append(epoch)
            return {"loss_epoch": [0.8, 0.2, 0.5][epoch]}

        monkeypatch.setattr(trainer, "train_epoch", train_epoch)
        monkeypatch.setattr(trainer, "val_epoch", val_epoch)
        if resume:
            assert (trainer.current_epoch, trainer.global_step) == (2, 2)
            assert trainer.best_val_loss == 0.2
        trainer.run_training()
        return trainer.checkpointer

    checkpointer = run(resume=False)
    assert (root / "interrupted").is_dir()
    assert checkpointer.read_best_epoch() == 1
    assert checkpointer.load_best_val_loss() == 0.2
    assert (root / "0").is_dir() == (not save_best)
    checkpointer = run(resume=True)
    assert trained == validated == [0, 1, 2]
    assert checkpointer.read_best_epoch() == 1
    steps = sorted(
        json.loads(p.read_text())["step"]
        for p in (root / "serving_eval").glob("*/evaluation.json")
    )
    assert steps == ([1, 2] if save_best else [1, 2, 3])


def test_disabled_and_interrupted_saves_never_run_evaluator(tmp_path, evaluator):
    trainer = _trainer(tmp_path)
    trainer.run_training()
    trainer.config = trainer.config._replace(checkpoint_eval_command=evaluator)
    trainer.maybe_save_checkpoint("interrupted")
    assert not (tmp_path / "serving_eval").exists()


def test_evaluation_failure_preserves_checkpoint_and_training_continues(
    tmp_path, caplog
):
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(7)")
    trainer = _trainer(tmp_path, f"{sys.executable} {script}")
    trainer.val_loader = None
    trainer.run_training()
    assert (tmp_path / "0" / "training_state.json").exists()
    assert "evaluation failed" in caplog.text.lower()
    assert not any(
        isinstance(r.msg, dict) and "serving_eval" in r.msg for r in caplog.records
    )


@pytest.mark.parametrize(
    "error", [ValueError("invalid metric name"), TrainingInterruptedError()]
)
def test_metric_backend_errors_are_isolated_but_training_interrupts_propagate(
    tmp_path, evaluator, caplog, error
):
    class FailingBackend(logging.Handler):
        def emit(self, record):
            if isinstance(record.msg, dict) and "serving_eval" in record.msg:
                raise error

    logger = logging.getLogger("speculators.metrics")
    handler = FailingBackend()
    logger.addHandler(handler)
    trainer = _trainer(tmp_path, evaluator, checkpoint_eval_mode="sync")
    trainer.maybe_save_checkpoint(0)
    try:
        with caplog.at_level(logging.INFO, logger="speculators.metrics"):
            if isinstance(error, TrainingInterruptedError):
                with pytest.raises(TrainingInterruptedError):
                    trainer._evaluate_checkpoint(0)
            else:
                trainer._evaluate_checkpoint(0)
                assert "evaluation failed" in caplog.text.lower()
        assert (tmp_path / "0" / "training_state.json").exists()
    finally:
        logger.removeHandler(handler)


def test_evaluator_config_resolves_from_cli_and_roundtrips():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "target",
            "--checkpoint-eval-command",
            "python eval.py",
            "--checkpoint-eval-timeout",
            "42",
            "--checkpoint-eval-steps",
            "250",
            "--checkpoint-eval-mode",
            "sync",
        ]
    )
    assert cfg.trainer.checkpoint_eval_command == "python eval.py"
    assert cfg.trainer.checkpoint_eval_timeout == 42
    assert cfg.trainer.checkpoint_eval_steps == 250
    assert cfg.trainer.checkpoint_eval_mode == "sync"
    assert TrainConfig.from_flat(cfg.flatten()).trainer == cfg.trainer


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_rejects_invalid_evaluation_timeout(timeout):
    with pytest.raises(SystemExit):
        TrainConfig.resolve(
            ["--verifier-name-or-path", "target", "--checkpoint-eval-timeout", timeout]
        )


@pytest.mark.parametrize("steps", ["0", "-5"])
def test_rejects_invalid_evaluation_steps(steps):
    with pytest.raises(SystemExit):
        TrainConfig.resolve(
            ["--verifier-name-or-path", "target", "--checkpoint-eval-steps", steps]
        )


def _distributed_failed_eval(rank, root):
    path = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        script = path / "invalid.py"
        trainer = _trainer(
            path, f"{sys.executable} {script}", checkpoint_eval_mode="sync"
        )
        trainer.rank = rank
        trainer.is_distributed = True
        trainer._evaluate_checkpoint(0)
        (path / f"rank{rank}.done").write_text("continued")
    finally:
        dist.destroy_process_group()


def test_distributed_invalid_result_does_not_abort_or_strand_peers(tmp_path):
    # JSON can express integers outside float64. This is an evaluator failure,
    # not a reason to abort rank 0 while the other ranks keep training.
    script = tmp_path / "invalid.py"
    script.write_text(
        "import os\nfrom pathlib import Path\n"
        "out = Path(os.environ['SPECULATORS_EVAL_OUTPUT_DIR'])\n"
        "out.joinpath('metrics.json').write_text('{\"mal\":' + '9'*400 + '}')\n"
    )
    mp.spawn(_distributed_failed_eval, args=(str(tmp_path),), nprocs=2)
    assert (tmp_path / "rank0.done").read_text() == "continued"
    assert (tmp_path / "rank1.done").read_text() == "continued"
    assert len(list((tmp_path / "serving_eval").glob("*/evaluation.json"))) == 1


def _distributed_evaluation(rank, root, ready):
    path = Path(root)
    # Complete imports on both spawned workers before starting the short
    # process-group timeout used to reproduce the original evaluation barrier.
    ready.wait(timeout=60)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    try:
        trainer = _trainer(
            path,
            f"{sys.executable} {path / 'evaluate.py'}",
            checkpoint_eval_mode="sync",
        )
        trainer.is_distributed = True
        trainer.rank = rank

        def save(_model, _optimizers, epoch):
            assert epoch == "interrupted"
            # Exercise the collective that an interrupt checkpoint would need.
            count = torch.ones(1)
            dist.all_reduce(count)
            assert count.item() == 2
            (path / f"rank{rank}.saved").touch()

        cast("Any", trainer.checkpointer).save_checkpoint = save

        @with_graceful_shutdown(timeout=10)
        def train(t):
            (path / f"rank{rank}.pid").write_text(str(os.getpid()))
            dist.barrier()
            t._evaluate_checkpoint(0)
            (path / f"rank{rank}.resumed").touch()
            dist.barrier()

        train(trainer)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "interrupt_rank", [None, 0, 1], ids=["long-eval", "signal-rank0", "signal-peer"]
)
def test_distributed_slow_evaluation_and_graceful_shutdown(tmp_path, interrupt_rank):
    script = tmp_path / "evaluate.py"
    root_pid = tmp_path / "rank0.pid"
    body = (
        "import os, signal, time\nfrom pathlib import Path\n"
        f"assert os.getppid() == int(Path({str(root_pid)!r}).read_text())\n"
    )
    if interrupt_rank is not None:
        pid_file = tmp_path / f"rank{interrupt_rank}.pid"
        body += (
            f"pid = int(Path({str(pid_file)!r}).read_text())\n"
            f"os.kill(pid, {int(signal.SIGTERM)})\n"
        )
    script.write_text(
        body + "time.sleep(4)\n"
        "out = Path(os.environ['SPECULATORS_EVAL_OUTPUT_DIR'])\n"
        "out.joinpath('metrics.json').write_text('{\"mal\": 2}')\n"
    )
    ready = mp.get_context("spawn").Barrier(2)
    mp.spawn(_distributed_evaluation, args=(str(tmp_path), ready), nprocs=2)
    for rank in (0, 1):
        assert (tmp_path / f"rank{rank}.saved").exists() == (interrupt_rank is not None)
        assert (tmp_path / f"rank{rank}.resumed").exists() == (interrupt_rank is None)
    metrics = list((tmp_path / "serving_eval").glob("*/metrics.json"))
    assert len(metrics) == (1 if interrupt_rank is None else 0)


def _distributed_done_interrupt_race(rank, root, ready):
    path = Path(root)
    ready.wait(timeout=60)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path / 'rendezvous'}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=3),
    )
    real_sleep = distributed_mod.time.sleep
    try:
        trainer = _trainer(
            path,
            f"{sys.executable} {path / 'evaluate.py'}",
            checkpoint_eval_mode="sync",
        )
        trainer.is_distributed = True
        trainer.rank = rank

        def save(_model, _optimizers, epoch):
            assert epoch == "interrupted"
            count = torch.ones(1)
            dist.all_reduce(count)
            assert count.item() == 3
            (path / f"rank{rank}.saved").touch()

        cast("Any", trainer.checkpointer).save_checkpoint = save

        if rank == 2:
            assert dist.group.WORLD is not None
            store = dist.group.WORLD.get_group_store()
            result_key = "speculators/serving_eval/result/1"
            ack_key = "speculators/serving_eval/ack/1"
            signaled = False

            def interrupt_after_done(seconds: Any) -> None:
                nonlocal signaled
                if not signaled:
                    (path / "rank2.waiting").touch()
                    while not store.check([result_key]):
                        real_sleep(0.01)
                    assert store.get(result_key) == b"done"
                    # Let rank 0 publish b"done" and rank 1 acknowledge before
                    # rank 2 is interrupted prior to acknowledging completion.
                    ack_wait = time.monotonic() + 1.0
                    while time.monotonic() < ack_wait and (
                        not store.check([ack_key]) or int(store.get(ack_key)) < 2
                    ):
                        real_sleep(0.01)
                    signaled = True
                    os.kill(os.getpid(), signal.SIGTERM)
                real_sleep(seconds)

            cast("Any", distributed_mod.time).sleep = interrupt_after_done

        @with_graceful_shutdown(timeout=10)
        def train(t):
            dist.barrier()
            t._evaluate_checkpoint(0)
            (path / f"rank{rank}.resumed").touch()
            dist.barrier()

        train(trainer)
    finally:
        distributed_mod.time.sleep = real_sleep
        dist.destroy_process_group()


def test_distributed_completion_interrupt_race_keeps_all_ranks_in_sync(tmp_path):
    script = tmp_path / "evaluate.py"
    waiting = tmp_path / "rank2.waiting"
    script.write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"waiting = Path({str(waiting)!r})\n"
        "deadline = time.monotonic() + 10\n"
        "while not waiting.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "assert waiting.exists()\n"
        "out = Path(os.environ['SPECULATORS_EVAL_OUTPUT_DIR'])\n"
        "out.joinpath('metrics.json').write_text('{\"mal\": 2}')\n"
    )
    ready = mp.get_context("spawn").Barrier(3)
    mp.spawn(_distributed_done_interrupt_race, args=(str(tmp_path), ready), nprocs=3)
    for rank in (0, 1, 2):
        assert (tmp_path / f"rank{rank}.saved").exists()
        assert not (tmp_path / f"rank{rank}.resumed").exists()


def test_async_eval_runs_without_blocking_training_and_survives_pruning(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "checkpoints"
    eval0_started = tmp_path / "eval0.started"
    epoch1_pruned = tmp_path / "epoch1.pruned"
    script = tmp_path / "async_eval.py"
    script.write_text(
        f"""
import json, os, time
from pathlib import Path
step = int(os.environ["SPECULATORS_EVAL_STEP"])
ckpt = Path(os.environ["SPECULATORS_EVAL_CHECKPOINT"])
if step == 1:
    Path({str(eval0_started)!r}).touch()
    # Stay active until epoch 1 trains AND prunes checkpoints/0 on disk.
    deadline = time.monotonic() + 10
    while not Path({str(epoch1_pruned)!r}).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert Path({str(epoch1_pruned)!r}).exists()
assert (ckpt / "config.json").exists()
out = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
(out / "metrics.json").write_text(json.dumps({{"mal": float(step) + 0.5}}))
"""
    )

    trainer = _trainer(
        root,
        f'"{sys.executable}" "{script}"',
        save_best=True,
        checkpoint_eval_mode="async",
    )
    trainer.config = trainer.config._replace(num_epochs=2)
    trainer.global_step = 0

    def train_epoch(epoch):
        if epoch == 1:
            # Wait briefly for epoch 0's background evaluator to start; epoch 1
            # begins even though epoch 0's evaluation has not finished yet.
            deadline = time.monotonic() + 10
            while not eval0_started.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert eval0_started.exists()
        trainer.global_step += 1

    def val_epoch(epoch):
        return {"loss_epoch": [0.8, 0.2][epoch]}

    orig_cleanup = trainer.checkpointer.cleanup_keep_only_best

    def cleanup_and_mark(best_epoch):
        orig_cleanup(best_epoch)
        if best_epoch == 1:
            assert not (root / "0").exists()
            epoch1_pruned.touch()

    monkeypatch.setattr(trainer, "train_epoch", train_epoch)
    monkeypatch.setattr(trainer, "val_epoch", val_epoch)
    monkeypatch.setattr(
        trainer.checkpointer, "cleanup_keep_only_best", cleanup_and_mark
    )

    with caplog.at_level(logging.INFO, logger="speculators.metrics"):
        trainer.run_training()

    records = [
        (r.step, r.msg["serving_eval"])
        for r in caplog.records
        if isinstance(r.msg, dict) and "serving_eval" in r.msg
    ]
    assert records == [(1, {"mal": 1.5}), (2, {"mal": 2.5})]
    snapshots = root / "serving_eval" / "_snapshots"
    assert not snapshots.exists() or not any(snapshots.iterdir())


def test_weights_only_step_eval_exports_model_without_optimizer(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "checkpoints"
    script = tmp_path / "step_eval.py"
    script.write_text(
        """
import json, os
from pathlib import Path
ckpt = Path(os.environ["SPECULATORS_EVAL_CHECKPOINT"])
assert (ckpt / "model.safetensors").exists()
assert not (ckpt / "optimizer_state_dict.pt").exists()
cfg = json.loads((ckpt / "config.json").read_text())
step = int(os.environ["SPECULATORS_EVAL_STEP"])
assert cfg["step"] == step
out = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
(out / "metrics.json").write_text(json.dumps({"mal": float(step)}))
"""
    )

    trainer = _trainer(
        root,
        f'"{sys.executable}" "{script}"',
        checkpoint_eval_steps=2,
        checkpoint_eval_mode="async",
    )
    trainer.config = trainer.config._replace(num_epochs=1)
    trainer.global_step = 0
    save_model_calls: list[int] = []
    orig_save_model = trainer.checkpointer.save_model

    def track_save_model(model, output_dir, float_dtype=torch.bfloat16):
        save_model_calls.append(trainer.global_step)
        orig_save_model(model, output_dir, float_dtype)

    monkeypatch.setattr(trainer.checkpointer, "save_model", track_save_model)

    def fake_train_epoch(epoch):
        for _ in range(4):
            trainer.global_step += 1
            trainer._flush_checkpoint_eval_results()
            if (
                trainer.config.checkpoint_eval_command
                and trainer.config.checkpoint_eval_steps is not None
                and trainer.global_step % trainer.config.checkpoint_eval_steps == 0
            ):
                trainer._evaluate_checkpoint(epoch, weights_only=True)

    monkeypatch.setattr(trainer, "train_epoch", fake_train_epoch)

    with caplog.at_level(logging.INFO, logger="speculators.metrics"):
        trainer.run_training()

    assert save_model_calls == [2, 4]
    records = [
        (r.step, r.msg["serving_eval"])
        for r in caplog.records
        if isinstance(r.msg, dict) and "serving_eval" in r.msg
    ]
    assert records == [(2, {"mal": 2.0}), (4, {"mal": 4.0})]
    # Decoupled step eval must not re-evaluate at epoch end.
    assert len(list((root / "serving_eval").glob("step_*/metrics.json"))) == 2
    snapshots = root / "serving_eval" / "_snapshots"
    assert not snapshots.exists() or not any(snapshots.iterdir())


def _distributed_async_sigterm(rank, root, ready):
    path = Path(root)
    ready.wait(timeout=60)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    try:
        trainer = _trainer(
            path,
            f"{sys.executable} {path / 'evaluate.py'}",
            checkpoint_eval_mode="async",
        )
        trainer.is_distributed = True
        trainer.rank = rank
        trainer.config = trainer.config._replace(num_epochs=2)
        trainer.global_step = 0

        def save(_model, _optimizers, epoch):
            if epoch == "interrupted":
                count = torch.ones(1)
                dist.all_reduce(count)
                assert count.item() == 2
                (path / f"rank{rank}.saved").touch()
                return
            out = path / str(epoch)
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")

        cast("Any", trainer.checkpointer).save_checkpoint = save

        def train_epoch(epoch):
            trainer.global_step += 1
            if epoch == 1:
                deadline = time.monotonic() + 10
                while not (path / "child.pid").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert (path / "child.pid").exists()
                dist.barrier()
                os.kill(os.getpid(), signal.SIGTERM)

        cast("Any", trainer).train_epoch = train_epoch
        trainer.run_training()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_distributed_async_sigterm_cancels_eval_and_saves_interrupted(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "evaluate.py"
    script.write_text(
        f"""
import subprocess, sys, time
from pathlib import Path
code = (
    "import signal, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
)
child = subprocess.Popen([sys.executable, "-c", code])
Path({str(child_pid_file)!r}).write_text(str(child.pid))
time.sleep(60)
"""
    )
    ready = mp.get_context("spawn").Barrier(2)
    mp.spawn(_distributed_async_sigterm, args=(str(tmp_path), ready), nprocs=2)

    for rank in (0, 1):
        assert (tmp_path / f"rank{rank}.saved").exists()
    pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"distributed async cancel leaked worker {pid}")


def test_snapshot_interrupt_propagates_and_cleans_partial_snapshot(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoints"
    trainer = _trainer(
        root,
        f"{sys.executable} -c pass",
        checkpoint_eval_mode="async",
    )
    trainer.global_step = 3
    (root / "0").mkdir(parents=True)
    (root / "0" / "a.txt").write_text("ok")
    (root / "0" / "b.safetensors").write_bytes(b"weights")
    saved_interrupted: list[int | str] = []
    orig_save = trainer.checkpointer.save_checkpoint

    def track_save(model, optimizers, epoch):
        saved_interrupted.append(epoch)
        orig_save(model, optimizers, epoch)

    monkeypatch.setattr(trainer.checkpointer, "save_checkpoint", track_save)
    monkeypatch.setattr(trainer, "train_epoch", lambda _epoch: None)
    monkeypatch.setattr(trainer, "val_epoch", lambda _epoch: None)

    orig_copy2 = checkpoint_eval_mod.shutil.copy2
    copied = 0

    def interrupt_on_second(src, dst, *args, **kwargs):
        nonlocal copied
        copied += 1
        if copied == 2:
            raise TrainingInterruptedError("SIGTERM during snapshot")
        return orig_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(checkpoint_eval_mod.shutil, "copy2", interrupt_on_second)

    trainer.run_training()

    assert "interrupted" in saved_interrupted
    assert not (root / "serving_eval" / "_snapshots").exists()


def test_snapshot_io_error_warns_cleans_partial_snapshot_and_continues(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "checkpoints"
    trainer = _trainer(
        root,
        f"{sys.executable} -c pass",
        checkpoint_eval_mode="async",
    )
    trainer.global_step = 3
    (root / "0").mkdir(parents=True)
    (root / "0" / "a.txt").write_text("ok")
    (root / "0" / "b.safetensors").write_bytes(b"weights")
    orig_copy2 = checkpoint_eval_mod.shutil.copy2
    copied = 0

    def fail_on_second(src, dst, *args, **kwargs):
        nonlocal copied
        copied += 1
        if copied == 2:
            raise OSError("disk full")
        return orig_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(checkpoint_eval_mod.shutil, "copy2", fail_on_second)

    with caplog.at_level(logging.WARNING, logger="speculators"):
        trainer._evaluate_checkpoint(0)

    assert any("snapshot failed" in r.getMessage() for r in caplog.records)
    assert not (root / "serving_eval" / "_snapshots").exists()


def _log_record(payload: dict[str, Any], step: int | None) -> logging.LogRecord:
    rec = logging.LogRecord(
        "speculators.metrics", logging.INFO, "", 0, payload, None, None
    )
    if step is not None:
        rec.step = step
    return rec


def test_wandb_handler_preserves_out_of_order_async_serving_eval_step():
    class FakeRun:
        def __init__(self):
            self.config: dict[str, Any] = {}
            self.defined: list[tuple[str, str | None, bool | None]] = []
            self.logged: list[tuple[dict[str, Any], int | None]] = []

        def define_metric(
            self,
            name: str,
            *,
            step_metric: str | None = None,
            step_sync: bool | None = None,
        ) -> None:
            self.defined.append((name, step_metric, step_sync))

        def log(self, data: dict[str, Any], step: int | None = None) -> None:
            self.logged.append((dict(data), step))

    handler = WandbHandler()
    fake_run = FakeRun()
    handler._run = cast("Any", fake_run)

    handler.emit(_log_record({"train": {"loss": 1.0}, "epoch": 1}, 20))
    handler.emit(_log_record({"serving_eval": {"gsm8k/mal": 2.0}, "epoch": 0}, 10))
    handler.emit(_log_record({"serving_eval": {"gsm8k/mal": 3.0}, "epoch": 0}, 15))
    handler.emit(_log_record({"val": {"loss": 0.5}, "epoch": 1}, 20))
    handler.emit(_log_record({"train": {"loss": 0.9}, "epoch": 1}, 21))
    handler.emit(_log_record({"serving_eval": {"gsm8k/mal": 3.5}, "epoch": 1}, 21))

    assert fake_run.defined == [
        ("trainer/step", None, None),
        ("*", "trainer/step", False),
        ("serving_eval/step", None, None),
        ("serving_eval/*", "serving_eval/step", False),
    ]
    assert fake_run.logged == [
        ({"train/loss": 1.0, "epoch": 1, "trainer/step": 20}, None),
        ({"serving_eval/gsm8k/mal": 2.0, "serving_eval/step": 10}, None),
        ({"serving_eval/gsm8k/mal": 3.0, "serving_eval/step": 15}, None),
        ({"val/loss": 0.5, "epoch": 1, "trainer/step": 20}, None),
        ({"train/loss": 0.9, "epoch": 1, "trainer/step": 21}, None),
        ({"serving_eval/gsm8k/mal": 3.5, "serving_eval/step": 21}, None),
    ]

    # Backends without define_metric (e.g. Trackio) keep explicit step logging.
    class LegacyRun:
        def __init__(self):
            self.logged: list[tuple[dict[str, Any], int | None]] = []

        def log(self, data: dict[str, Any], step: int | None = None) -> None:
            self.logged.append((dict(data), step))

    legacy = WandbHandler()
    legacy_run = LegacyRun()
    legacy._run = cast("Any", legacy_run)
    legacy.emit(_log_record({"train": {"loss": 1.0}}, 7))
    assert legacy_run.logged == [({"train/loss": 1.0}, 7)]


def _iter_wandb_record_bytes(data: bytes):
    assert data[:4] == b":W&B"
    pos, block = 7, 32768
    buf = bytearray()
    while pos + 7 <= len(data):
        rem = block - (pos % block)
        if rem < 7:
            pos += rem
            continue
        _, length, rtype = struct.unpack("<IHB", data[pos : pos + 7])
        pos += 7
        chunk = data[pos : pos + length]
        pos += length
        if rtype == 1:
            yield chunk
        elif rtype == 2:
            buf = bytearray(chunk)
        elif rtype == 3:
            buf.extend(chunk)
        elif rtype == 4:
            buf.extend(chunk)
            yield bytes(buf)
            buf = bytearray()
        elif rtype == 0:
            pos += block - (pos % block)


def _parse_wandb_offline_history(wandb_file: Path) -> list[dict[str, Any]]:
    pb = importlib.import_module("wandb.proto.wandb_internal_pb2")
    rows: list[dict[str, Any]] = []
    for payload in _iter_wandb_record_bytes(wandb_file.read_bytes()):
        rec = pb.Record()
        rec.ParseFromString(payload)
        if rec.HasField("history"):
            row: dict[str, Any] = {}
            for item in rec.history.item:
                key = item.key or "/".join(item.nested_key)
                if not key.startswith("_"):
                    row[key] = json.loads(item.value_json)
            rows.append(row)
    return rows


def test_wandb_handler_offline_run_preserves_out_of_order_serving_eval(tmp_path):
    wandb = pytest.importorskip("wandb")

    handler = WandbHandler(
        run_name="async-serving-eval-offline",
        log_dir=tmp_path,
        project="speculators-test",
        mode="offline",
        settings=wandb.Settings(silent=True),
    )
    run = handler._setup()
    handler._run = run
    run_dir = Path(run.dir)
    try:
        handler.emit(_log_record({"train": {"loss": 1.0}, "epoch": 1}, 20))
        handler.emit(_log_record({"serving_eval": {"gsm8k/mal": 2.0}, "epoch": 0}, 10))
        handler.emit(_log_record({"serving_eval": {"gsm8k/mal": 3.0}, "epoch": 0}, 15))
        handler.emit(_log_record({"val": {"loss": 0.5}, "epoch": 1}, 20))
        handler.emit(_log_record({"train": {"loss": 0.9}, "epoch": 1}, 21))
        handler.emit(_log_record({"serving_eval": {"gsm8k/mal": 3.5}, "epoch": 1}, 21))
    finally:
        run.finish()

    for log_name in ("debug.log", "debug-internal.log"):
        log_path = run_dir / "logs" / log_name
        if log_path.exists():
            text = log_path.read_text()
            assert "Steps must be monotonically increasing" not in text

    wandb_file = next(run_dir.parent.glob("*.wandb"))
    rows = _parse_wandb_offline_history(wandb_file)
    assert rows == [
        {"train/loss": 1, "epoch": 1, "trainer/step": 20},
        {"serving_eval/gsm8k/mal": 2, "serving_eval/step": 10},
        {"serving_eval/gsm8k/mal": 3, "serving_eval/step": 15},
        {"val/loss": 0.5, "epoch": 1, "trainer/step": 20},
        {"train/loss": 0.9, "epoch": 1, "trainer/step": 21},
        {"serving_eval/gsm8k/mal": 3.5, "serving_eval/step": 21},
    ]


def test_single_gpu_checkpointer_save_model_writes_weights_only(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))

        def save_pretrained(self, save_directory, state_dict=None):
            out = Path(save_directory)
            out.mkdir(parents=True, exist_ok=True)
            assert state_dict is not None
            assert state_dict["weight"].dtype == torch.bfloat16
            (out / "model.safetensors").write_bytes(b"bf16-weights")
            (out / "config.json").write_text(json.dumps({"dtype": "float32"}))

    cp = SingleGPUCheckpointer(tmp_path)
    out = tmp_path / "serving_eval" / "_snapshots" / "step_4"
    cp.save_model(cast("Any", TinyModel()), out)

    assert (out / "model.safetensors").exists()
    assert json.loads((out / "config.json").read_text())["dtype"] == "bfloat16"
    assert not (out / "optimizer_state_dict.pt").exists()
    assert not (out / "scheduler_state_dict.pt").exists()


def _distributed_save_model_worker(rank, root, ready):
    path = Path(root)
    ready.wait(timeout=60)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        orig_get_rank = checkpointer_mod.get_rank
        checkpointer_mod.get_rank = dist.get_rank
        try:

            class TinyDistributedModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.full((2, 2), float(rank + 1), dtype=torch.float32)
                    )

                def save_pretrained(self, save_directory, state_dict=None):
                    assert rank == 0
                    out_dir = Path(save_directory)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    assert state_dict is not None
                    assert state_dict["weight"].dtype == torch.bfloat16
                    (out_dir / "model.safetensors").write_bytes(b"dist-bf16-weights")
                    (out_dir / "config.json").write_text(
                        json.dumps({"dtype": "float32"})
                    )

            cp = DistributedCheckpointer(path)
            out = path / "serving_eval" / "_snapshots" / "step_8"
            cp.save_model(cast("Any", TinyDistributedModel()), out)
            (path / f"rank{rank}.done").touch()
        finally:
            checkpointer_mod.get_rank = orig_get_rank
    finally:
        dist.destroy_process_group()


def test_distributed_checkpointer_save_model_writes_weights_only_on_rank0(tmp_path):
    ready = mp.get_context("spawn").Barrier(2)
    mp.spawn(_distributed_save_model_worker, args=(str(tmp_path), ready), nprocs=2)

    out = tmp_path / "serving_eval" / "_snapshots" / "step_8"
    for rank in (0, 1):
        assert (tmp_path / f"rank{rank}.done").exists()
    assert (out / "model.safetensors").exists()
    assert json.loads((out / "config.json").read_text())["dtype"] == "bfloat16"
    assert not (out / "optimizer_state_dict.pt").exists()
    assert not (out / "scheduler_state_dict.pt").exists()
