"""Periodic re-synchronisation inside the validation loop."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from speculators.train.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Unit tests for _maybe_val_sync helper
# ---------------------------------------------------------------------------


def _barrier_indices(n_batches: int, *, is_distributed: bool, interval: int) -> list:
    stand_in = cast("Trainer", SimpleNamespace(is_distributed=is_distributed))
    hit = []
    with (
        patch("speculators.train.trainer._VAL_SYNC_INTERVAL", interval),
        patch("speculators.train.trainer.dist.barrier") as barrier,
    ):
        for i in range(n_batches):
            before = barrier.call_count
            Trainer._maybe_val_sync(stand_in, i)
            if barrier.call_count > before:
                hit.append(i)
    return hit


def test_barriers_on_interval_boundaries_only():
    assert _barrier_indices(200, is_distributed=True, interval=50) == [50, 100, 150]


def test_never_barriers_on_the_first_batch():
    assert 0 not in _barrier_indices(10, is_distributed=True, interval=1)


def test_no_barrier_when_single_process():
    assert _barrier_indices(200, is_distributed=False, interval=50) == []


def test_interval_zero_disables_syncing():
    assert _barrier_indices(200, is_distributed=True, interval=0) == []


@pytest.mark.parametrize("interval", [1, 7, 50, 500])
def test_barrier_count_is_deterministic_for_a_given_batch_count(interval):
    n = 1864
    first = _barrier_indices(n, is_distributed=True, interval=interval)
    second = _barrier_indices(n, is_distributed=True, interval=interval)
    assert first == second
    assert len(first) == (n - 1) // interval


# ---------------------------------------------------------------------------
# Integration: _maybe_val_sync is actually called through val_epoch
# ---------------------------------------------------------------------------


def _fake_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "hidden_states": torch.zeros(1, 4, 16),
    }


def _make_val_trainer(n_batches: int, *, is_distributed: bool) -> "Trainer":
    model = MagicMock()
    model.return_value = (
        None,
        torch.tensor(0.5),
        {"loss": torch.tensor(0.5)},
    )

    batches = [_fake_batch() for _ in range(n_batches)]
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter(batches))
    loader.__len__ = MagicMock(return_value=len(batches))
    loader.batch_sampler = MagicMock(spec=[])

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.val_loader = loader
    trainer.is_distributed = is_distributed
    trainer.rank = 1
    trainer.local_rank = 0
    trainer.device_type = "cpu"
    trainer.global_step = 0
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=1,
        save_path="/tmp",
        hidden_states_dtype=torch.float32,
        val_call_kwargs=None,
    )
    return trainer


@pytest.mark.parametrize("n_batches", [120, 200])
def test_val_epoch_calls_barrier_at_sync_interval(n_batches):
    trainer = _make_val_trainer(n_batches, is_distributed=True)
    with (
        patch("speculators.train.trainer._VAL_SYNC_INTERVAL", 50),
        patch("speculators.train.trainer.dist.barrier") as barrier,
        patch("speculators.train.trainer.dist.all_reduce"),
        patch("speculators.train.trainer.dist.get_world_size", return_value=1),
    ):
        trainer.val_epoch(0)
        assert barrier.call_count == (n_batches - 1) // 50


def test_val_epoch_no_barrier_when_not_distributed():
    trainer = _make_val_trainer(120, is_distributed=False)
    with (
        patch("speculators.train.trainer._VAL_SYNC_INTERVAL", 50),
        patch("speculators.train.trainer.dist.barrier") as barrier,
    ):
        trainer.val_epoch(0)
        barrier.assert_not_called()


def test_val_epoch_empty_loader_returns_empty_metrics():
    trainer = _make_val_trainer(0, is_distributed=False)
    result = trainer.val_epoch(0)
    assert result == {}
