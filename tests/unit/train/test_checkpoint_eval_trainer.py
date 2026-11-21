"""Checkpoint-boundary tests without loading or training a language model."""

import json
import logging
import os
import signal
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.config import TrainConfig
from speculators.train.graceful_shutdown import (
    TrainingInterruptedError,
    with_graceful_shutdown,
)
from speculators.train.trainer import Trainer, TrainerConfig


def _trainer(tmp_path: Path, command: str | None = None, *, save_best=False) -> Trainer:
    trainer = cast("Any", object.__new__(Trainer))
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=1,
        save_path=str(tmp_path),
        checkpoint_eval_command=command,
        checkpoint_eval_timeout=10,
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

    def save(_model, _optimizers, epoch):
        path = tmp_path / str(epoch)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")

    trainer.checkpointer = SingleGPUCheckpointer(tmp_path)
    trainer.checkpointer.save_checkpoint = save
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
        trainer = _trainer(root, f'"{sys.executable}" "{script}"', save_best=save_best)
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
    trainer = _trainer(tmp_path, evaluator)
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
        ]
    )
    assert cfg.trainer.checkpoint_eval_command == "python eval.py"
    assert cfg.trainer.checkpoint_eval_timeout == 42
    assert TrainConfig.from_flat(cfg.flatten()).trainer == cfg.trainer


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_rejects_invalid_evaluation_timeout(timeout):
    with pytest.raises(SystemExit):
        TrainConfig.resolve(
            ["--verifier-name-or-path", "target", "--checkpoint-eval-timeout", timeout]
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
        trainer = _trainer(path, f"{sys.executable} {script}")
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
        trainer = _trainer(path, f"{sys.executable} {path / 'evaluate.py'}")
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
