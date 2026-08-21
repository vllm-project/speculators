"""Unit tests for accelerator-agnostic seeding and cache release in scripts/train.py.

The script is not a package, so it is imported by path.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Add scripts/ to the import path the same way the other script tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from train import _accelerator_module, set_seed  # type: ignore[import-not-found]


def test_accelerator_module_is_none_without_accelerator(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    assert _accelerator_module() is None


def test_set_seed_without_accelerator_is_reproducible(monkeypatch):
    """Seeding must not require torch.cuda when no accelerator is present."""
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    assert torch.equal(first, torch.rand(4))


def test_set_seed_seeds_current_accelerator_module(monkeypatch):
    """The device module of the current accelerator is seeded, not torch.cuda."""
    calls: list[int] = []
    fake_module = SimpleNamespace(manual_seed_all=calls.append)
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: SimpleNamespace(type="npu")
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device_type: fake_module)
    set_seed(7)
    assert calls == [7]
