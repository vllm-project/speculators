from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from torch import nn

from speculators.train.data import (
    WINDOWED_BATCH_LEASES_KEY,
    WINDOWED_LEASE_KEY,
    create_collate_fn,
)
from speculators.train.dataloader import WindowedBatchSampler
from speculators.train.trainer import Trainer, TrainerConfig


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Sampler:
    epoch = 0
    seed = 7
    rank = 0
    num_replicas = 1
    batch_max_length = 16
    lengths = [2, 2, 2]

    def __init__(self) -> None:
        self.batches = {0: [[2, 0], [1]], 1: [[1], [0, 2]]}

    def _generate_batches(self, epoch: int) -> list[list[int]]:
        return self.batches[epoch]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_windowed_batch_sampler_positions_are_stable_across_epochs():
    first = WindowedBatchSampler(
        _Sampler(),
        stream_id=_digest("stream"),
        request_id_for_index=lambda index: _digest(f"request-{index}"),
    )
    second = WindowedBatchSampler(
        _Sampler(),
        stream_id=_digest("stream"),
        request_id_for_index=lambda index: _digest(f"request-{index}"),
    )

    epoch_zero = first._generate_batches(0)
    epoch_one = first._generate_batches(1)
    repeated = second._generate_batches(1)

    assert [sample.dataset_index for batch in epoch_zero for sample in batch] == [
        2,
        0,
        1,
    ]
    assert [sample.sequence for batch in epoch_one for sample in batch] == [3, 4, 5]
    assert [sample.position_id for batch in epoch_one for sample in batch] == [
        sample.position_id for batch in repeated for sample in batch
    ]
    assert [
        (sample.batch_start_sequence, sample.batch_end_sequence)
        for sample in epoch_one[1]
    ] == [(4, 6), (4, 6)]


def _sample(lease: dict[str, Any] | None = None) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "hidden_states": torch.zeros(2, 4),
        "input_ids": torch.tensor([1, 2]),
        "verifier_last_hidden_states": torch.zeros(2, 2),
        "loss_mask": torch.ones(2),
        "lengths": torch.tensor([2]),
        "position_ids": torch.arange(2),
    }
    if lease is not None:
        sample[WINDOWED_LEASE_KEY] = lease
    return sample


def test_collate_keeps_artifact_leases_out_of_model_tensors():
    lease = {
        "token": "lease",
        "consumer_id": "consumer",
        "stream_id": _digest("stream"),
        "sequence": 0,
        "request_id": _digest("request"),
        "generation": 0,
    }
    collate = create_collate_fn(max_len=4, hidden_size=2, num_target_layers=1)

    batch = collate([_sample(lease)])

    assert batch.pop(WINDOWED_BATCH_LEASES_KEY) == [lease]
    assert WINDOWED_LEASE_KEY not in batch
    assert all(isinstance(value, torch.Tensor) for value in batch.values())


@dataclass
class _RecordingDataset:
    events: list[str]
    stop_error: Exception | None = None

    def ack_windowed_batch(self, _leases: list[dict[str, Any]]) -> None:
        self.events.append("ack")

    def abandon_windowed_batch(self, _leases: list[dict[str, Any]]) -> None:
        self.events.append("abandon")

    def stop_windowed_producer(self, *, completed: bool = False) -> None:
        self.events.append(f"stop:{completed}")
        if self.stop_error is not None:
            raise self.stop_error


class _Loader:
    def __init__(self, dataset: _RecordingDataset, batch: dict[str, Any]) -> None:
        self.dataset = dataset
        self.batch_sampler = object()
        self.batch = batch

    def __iter__(self):
        yield self.batch

    def __len__(self) -> int:
        return 1


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, **_batch):
        loss = self.weight.square()
        return None, loss, {"loss": loss.detach()}


def _trainer(events: list[str]) -> Trainer:
    lease = {
        "token": "lease",
        "consumer_id": "consumer",
        "stream_id": _digest("stream"),
        "sequence": 0,
        "request_id": _digest("request"),
        "generation": 0,
    }
    batch = {
        "input_ids": torch.tensor([[1]]),
        "document_ids": torch.tensor([[0]]),
        WINDOWED_BATCH_LEASES_KEY: [lease],
    }
    trainer = cast("Any", Trainer.__new__(Trainer))
    trainer.model = _Model()
    trainer.config = TrainerConfig(
        lr=0.1,
        num_epochs=1,
        save_path="unused",
        hidden_states_dtype=torch.bfloat16,
        log_freq=100,
        scheduler_type="none",
    )
    trainer.local_rank = torch.device("cpu")
    trainer.rank = 1
    trainer.is_distributed = False
    trainer.device_type = "cpu"
    trainer.train_loader = _Loader(_RecordingDataset(events), batch)
    trainer.global_step = 1
    trainer.current_epoch = 0
    trainer._resume_local_step = 0
    trainer._prepared_windowed_datasets = set()
    trainer.optimizers = [torch.optim.SGD(trainer.model.parameters(), lr=0.1)]
    trainer.schedulers = []
    return trainer


def test_trainer_acks_only_after_optimizer_step(monkeypatch):
    events: list[str] = []
    trainer = _trainer(events)
    original_step = trainer._optimizers_step

    def step() -> None:
        original_step()
        events.append("optimizer")

    monkeypatch.setattr(trainer, "_optimizers_step", step)
    trainer.train_epoch(0)

    assert events == ["optimizer", "ack"]


def test_optimizer_failure_abandons_without_ack(monkeypatch):
    events: list[str] = []
    trainer = _trainer(events)

    def fail() -> None:
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(trainer, "_optimizers_step", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        trainer.train_epoch(0)

    assert events == ["abandon"]


def test_windowed_phase_completes_consumer_after_success():
    events: list[str] = []
    loader = _Loader(_RecordingDataset(events), {})

    def operation(epoch: int) -> str:
        events.append(f"run:{epoch}")
        return "result"

    assert Trainer._run_windowed_phase(cast("Any", loader), operation, 3) == "result"
    assert events == ["run:3", "stop:True"]


def test_windowed_phase_stops_without_completion_after_failure():
    events: list[str] = []
    loader = _Loader(_RecordingDataset(events), {})

    def operation(_epoch: int) -> None:
        events.append("run")
        raise RuntimeError("phase failed")

    with pytest.raises(RuntimeError, match="phase failed"):
        Trainer._run_windowed_phase(cast("Any", loader), operation, 0)
    assert events == ["run", "stop:False"]


def test_windowed_cleanup_error_does_not_replace_phase_failure(caplog):
    events: list[str] = []
    dataset = _RecordingDataset(events, stop_error=RuntimeError("producer failed"))
    loader = _Loader(dataset, {})

    def operation(_epoch: int) -> None:
        raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed"):
        Trainer._run_windowed_phase(cast("Any", loader), operation, 0)
    assert events == ["stop:False"]
    assert "cleanup failed" in caplog.text


def test_windowed_cleanup_error_propagates_after_success():
    events: list[str] = []
    dataset = _RecordingDataset(events, stop_error=RuntimeError("producer failed"))
    loader = _Loader(dataset, {})

    with pytest.raises(RuntimeError, match="producer failed"):
        Trainer._run_windowed_phase(cast("Any", loader), lambda _epoch: None, 0)
    assert events == ["stop:True"]
