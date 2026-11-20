"""Checkpoint-boundary tests without loading or training a language model."""

import logging
import os
import signal
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
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
    trainer.global_step = 17
    trainer.best_val_loss = float("inf")
    trainer.schedulers = []
    trainer.optimizers = []
    trainer.model = None
    trainer._last_checkpoint_eval = None

    def save(_model, _optimizers, epoch):
        path = tmp_path / str(epoch)
        path.mkdir(exist_ok=True)
        (path / "config.json").write_text("{}")

    trainer.checkpointer = SimpleNamespace(
        path=tmp_path,
        save_checkpoint=save,
        save_val_metrics=lambda *a: None,
        update_best_symlink=lambda *a: None,
        cleanup_keep_only_best=lambda *a, **kw: None,
    )
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
        trainer.maybe_save_checkpoint(0, local_step=3)
        trainer.maybe_save_checkpoint(0)
    results = list((tmp_path / "serving_eval").glob("*/metrics.json"))
    assert len(results) == 1
    records = [
        r for r in caplog.records if isinstance(r.msg, dict) and "serving_eval" in r.msg
    ]
    assert len(records) == 1
    assert records[0].msg["serving_eval"] == {"mean_acceptance_length": 3.25}
    assert records[0].step == 17


def test_best_only_checkpoint_also_evaluated(tmp_path, evaluator):
    trainer = _trainer(tmp_path, evaluator, save_best=True)
    trainer.maybe_save_checkpoint(0)
    assert not (tmp_path / "serving_eval").exists()
    trainer.maybe_update_best(0, {"loss_epoch": 0.5})
    assert len(list((tmp_path / "serving_eval").glob("*/metrics.json"))) == 1


def test_best_retention_preserves_metrics_and_failed_evaluation_logs(
    tmp_path, evaluator
):
    trainer = _trainer(tmp_path, evaluator, save_best=True)
    real_checkpointer = SingleGPUCheckpointer(tmp_path)
    cast(
        "Any", trainer.checkpointer
    ).cleanup_keep_only_best = real_checkpointer.cleanup_keep_only_best
    historical = tmp_path / "serving_eval" / "old"
    historical.mkdir(parents=True)
    (historical / "metrics.json").write_text('{"acceptance_length": 2}')
    failed = tmp_path / "serving_eval" / "failed"
    failed.mkdir()
    (failed / "evaluator.log").write_text("previous failure")

    trainer.maybe_update_best(0, {"loss_epoch": 0.5})

    assert (historical / "metrics.json").exists()
    assert (failed / "evaluator.log").read_text() == "previous failure"
    assert len(list((tmp_path / "serving_eval").glob("*/metrics.json"))) == 2


def test_disabled_and_interrupted_saves_never_run_evaluator(tmp_path, evaluator):
    trainer = _trainer(tmp_path)
    trainer.maybe_save_checkpoint(0)
    trainer.config = trainer.config._replace(checkpoint_eval_command=evaluator)
    trainer.maybe_save_checkpoint("interrupted")
    assert not (tmp_path / "serving_eval").exists()


def test_evaluation_failure_preserves_checkpoint_and_training_continues(
    tmp_path, caplog
):
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(7)")
    trainer = _trainer(tmp_path, f"{sys.executable} {script}")
    trainer.maybe_save_checkpoint(0)
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
    try:
        with caplog.at_level(logging.INFO, logger="speculators.metrics"):
            if isinstance(error, TrainingInterruptedError):
                with pytest.raises(TrainingInterruptedError):
                    trainer.maybe_save_checkpoint(0)
            else:
                trainer.maybe_save_checkpoint(0)
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
