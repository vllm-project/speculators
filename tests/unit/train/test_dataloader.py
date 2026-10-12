"""Unit tests for training dataloader setup."""

from unittest.mock import Mock

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
