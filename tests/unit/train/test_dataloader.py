"""Unit tests for training dataloader setup."""

import pickle
from unittest.mock import Mock

import pytest

from speculators.train import dataloader


def test_worker_init_binds_to_local_rank(monkeypatch):
    set_num_threads = Mock()
    is_available = Mock(return_value=True)
    set_device_index = Mock()

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(dataloader.torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(dataloader.torch.accelerator, "is_available", is_available)
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0)

    set_num_threads.assert_called_once_with(1)
    is_available.assert_called_once_with()
    set_device_index.assert_called_once_with(3)


def test_worker_init_skips_device_binding_without_accelerator(monkeypatch):
    set_num_threads = Mock()
    is_available = Mock(return_value=False)
    set_device_index = Mock()

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(dataloader.torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(dataloader.torch.accelerator, "is_available", is_available)
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0)

    set_num_threads.assert_called_once_with(1)
    is_available.assert_called_once_with()
    set_device_index.assert_not_called()


def test_worker_init_skips_device_binding_when_not_pinned(monkeypatch):
    """CPU-only backends must not select the rank's device: a non-zero
    ``set_device_index`` creates a CUDA context in every worker."""
    set_device_index = Mock()

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(dataloader.torch, "set_num_threads", Mock())
    monkeypatch.setattr(
        dataloader.torch.accelerator, "is_available", Mock(return_value=True)
    )
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0, pin_device=False)

    set_device_index.assert_not_called()


@pytest.mark.parametrize("pin_device", [True, False])
def test_setup_dataloader_forwards_pin_device_to_spawned_workers(
    monkeypatch, pin_device
):
    """The worker init must carry ``pin_device`` and survive pickling, since
    workers start with the ``spawn`` context."""
    monkeypatch.setattr(dataloader, "get_dp_size", lambda: 1)
    monkeypatch.setattr(dataloader, "get_dp_rank", lambda: 0)
    dataset = Mock(approx_lengths=[4, 4])

    loader = dataloader._setup_dataloader(
        dataset, total_seq_len=8, num_workers=2, pin_device=pin_device
    )

    worker_init_fn = pickle.loads(pickle.dumps(loader.worker_init_fn))  # noqa: S301
    assert worker_init_fn.func is dataloader._worker_init_fn
    assert worker_init_fn.keywords == {"pin_device": pin_device}
