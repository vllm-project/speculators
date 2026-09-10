import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from speculators.train.logger import FormatDictFilter, IsRank0Filter
from speculators.train.trainer import Trainer, TrainerConfig


def _record(**extra):
    record = logging.LogRecord(
        "speculators", logging.INFO, __file__, 0, "msg", None, None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


@pytest.fixture
def clean_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_global_rank0_filter_passes_only_global_rank0(monkeypatch, clean_rank_env):
    # Multi-node: a non-zero global rank that happens to be local_rank 0
    # must still be filtered out (the bug this guards against).
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert IsRank0Filter().filter(_record()) is False

    monkeypatch.setenv("RANK", "0")
    assert IsRank0Filter().filter(_record()) is True


def test_override_bypasses_filter(clean_rank_env, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert IsRank0Filter().filter(_record(override_rank0_filter=True)) is True


@pytest.mark.parametrize("split", ["train", "val"])
def test_reference_counts_reach_console_without_rounding(split):
    trainer = MagicMock(spec=Trainer)
    trainer.rank = 1
    trainer.local_rank = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.is_distributed = True
    trainer.global_step = 0
    trainer.config = TrainerConfig(
        lr=1e-3, num_epochs=1, save_path="/tmp", max_steps=1, log_freq=1
    )
    trainer.optimizers = [MagicMock(param_groups=[{"lr": 1e-3}])]
    trainer._prepare_resume_skip.return_value = 0
    trainer.model = MagicMock()
    trainer.model.parameters.return_value = []
    trainer.model.return_value = (
        None,
        torch.tensor(0.5, requires_grad=True),
        {
            "loss_sum": torch.tensor(0.5),
            "loss_total": torch.tensor(1.0),
            "reference_acc_at_pos_0_sum": torch.tensor(12345.0),
            "reference_acc_at_pos_0_total": torch.tensor(654321.0),
        },
    )
    loader = MagicMock()
    loader.__len__.return_value = 1
    loader.__iter__.return_value = iter(
        [{"document_ids": torch.zeros(1, 4, dtype=torch.long), "error_records": 0}]
    )
    trainer.train_loader = trainer.val_loader = loader

    with (
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch("speculators.train.trainer.dist.get_world_size", return_value=2),
        patch(
            "speculators.train.trainer.dist.reduce",
            side_effect=lambda x, **kw: x.mul_(2),
        ),
        patch(
            "speculators.train.trainer.dist.all_reduce",
            side_effect=lambda x, **kw: x.mul_(2),
        ),
        patch("speculators.train.trainer.metric_logger.info") as log,
    ):
        getattr(Trainer, f"{split}_epoch")(trainer, 0)

    record = _record()
    record.msg = log.call_args.args[0]
    suffix = "_epoch" if split == "val" else ""
    metrics = record.msg[split]
    assert metrics[f"reference_acc_at_pos_0{suffix}"] == pytest.approx(12345 / 654321)
    FormatDictFilter().filter(record)
    assert f"{split}/reference_acc_at_pos_0_sum{suffix}=24690" in record.msg
    assert f"{split}/reference_acc_at_pos_0_total{suffix}=1308642" in record.msg
    assert f"{split}/loss{suffix}=0.500" in record.msg


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    ("log_freq", "max_steps", "windows"),
    [
        (1, None, [[0], [1], [2], [3], [4]]),
        (3, None, [[0], [1, 2, 3], [4]]),
        (3, 3, [[0], [1, 2]]),
    ],
)
def test_training_metrics_pool_batches_between_logs(
    log_freq, max_steps, windows, world_size
):
    trainer = MagicMock(spec=Trainer)
    trainer.rank = 1
    trainer.local_rank = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.is_distributed = world_size > 1
    trainer.global_step = 0
    trainer.config = TrainerConfig(
        lr=1e-3, num_epochs=1, save_path="/tmp", log_freq=log_freq, max_steps=max_steps
    )
    trainer.optimizers = [MagicMock(param_groups=[{"lr": 1e-3}])]
    trainer._prepare_resume_skip.return_value = 0
    trainer.model = MagicMock()
    trainer.model.parameters.return_value = []
    # Unequal denominators distinguish pooled counts from averaged batch rates.
    counts = [(0, 1), (8, 10), (1, 2), (0, 0), (1, 1)]
    trainer.model.side_effect = [
        (
            None,
            torch.tensor(float(i + 1), requires_grad=True),
            {
                "loss_sum": torch.tensor(float(i + 1)),
                "loss_total": torch.tensor(1.0),
                "loss_step_0": torch.tensor(float(i + 1)),
                "full_acc_sum": torch.tensor(correct),
                "full_acc_total": torch.tensor(total),
                "reference_acc_at_pos_0_sum": torch.tensor(correct),
                "reference_acc_at_pos_0_total": torch.tensor(total),
            },
        )
        for i, (correct, total) in enumerate(counts)
    ]
    loader = MagicMock()
    loader.__len__.return_value = len(counts)
    loader.__iter__.return_value = iter(
        [
            {"document_ids": torch.zeros(1, 4, dtype=torch.long), "error_records": 0}
            for _ in counts
        ]
    )
    trainer.train_loader = loader

    with (
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch("speculators.train.trainer.dist.get_world_size", return_value=world_size),
        patch(
            "speculators.train.trainer.dist.reduce",
            side_effect=lambda x, **kw: x.mul_(world_size),
        ),
        patch("speculators.train.trainer.metric_logger.info") as log,
    ):
        Trainer.train_epoch(trainer, 0)

    assert trainer.model.call_count == sum(map(len, windows))
    assert log.call_count == len(windows)
    for call, window in zip(log.call_args_list, windows, strict=True):
        record = call.args[0]
        metrics = record["train"]
        correct = sum(counts[i][0] for i in window) * world_size
        total = sum(counts[i][1] for i in window) * world_size
        rate = correct / total if total else 0.0
        assert record["global_step"] == window[-1]
        assert record["lr"] == 1e-3
        assert metrics["reference_acc_at_pos_0_sum"] == correct
        assert metrics["reference_acc_at_pos_0_total"] == total
        assert metrics["reference_acc_at_pos_0"] == pytest.approx(rate)
        assert metrics["full_acc"] == pytest.approx(rate)
        assert metrics["loss"] == pytest.approx(
            sum(i + 1 for i in window) / len(window)
        )
        assert metrics["loss_step_0"] == metrics["loss"]
