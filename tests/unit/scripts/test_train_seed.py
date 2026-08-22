"""Unit tests for seeding and accelerator cache release in scripts/train.py.

The script is not a package, so it is imported by path.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Add scripts/ to the import path the same way the other script tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from train import (  # type: ignore[import-not-found]
    _accelerator_module,
    _release_accelerator_cache,
    set_seed,
)


def _no_accelerator(**_kwargs):
    return None


def _fake_accelerator(device_type: str):
    seen: dict[str, object] = {}

    def current_accelerator(**kwargs):
        seen["kwargs"] = kwargs
        return SimpleNamespace(type=device_type)

    return current_accelerator, seen


def test_accelerator_module_is_none_without_accelerator(monkeypatch):
    """No runtime accelerator means no device module (and no torch.cuda call)."""
    monkeypatch.setattr(torch.accelerator, "current_accelerator", _no_accelerator)
    assert _accelerator_module() is None


def test_accelerator_module_checks_runtime_availability(monkeypatch):
    """The accelerator is resolved with check_available=True, not compile-time only."""
    current_accelerator, seen = _fake_accelerator("npu")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", current_accelerator)
    monkeypatch.setattr(torch, "get_device_module", lambda _t: SimpleNamespace())
    _accelerator_module()
    assert seen["kwargs"] == {"check_available": True}


def test_set_seed_is_reproducible_and_device_agnostic():
    """Seeding relies on torch.manual_seed (which seeds every device generator)."""
    assert "torch.cuda" not in inspect.getsource(set_seed)
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    assert torch.equal(first, torch.rand(4))


def test_release_accelerator_cache_calls_device_module(monkeypatch):
    """Cache release goes through the current accelerator's empty_cache."""
    calls: list[str] = []
    requested: list[str] = []
    fake_module = SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))

    def get_device_module(device_type):
        requested.append(device_type)
        return fake_module

    current_accelerator, _ = _fake_accelerator("npu")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", current_accelerator)
    monkeypatch.setattr(torch, "get_device_module", get_device_module)
    _release_accelerator_cache()
    assert requested == ["npu"]
    assert calls == ["empty_cache"]


def test_release_accelerator_cache_is_noop_without_accelerator(monkeypatch):
    """Without an accelerator the cache release is skipped entirely."""
    monkeypatch.setattr(torch.accelerator, "current_accelerator", _no_accelerator)
    monkeypatch.setattr(
        torch, "get_device_module", lambda _t: (_ for _ in ()).throw(AssertionError)
    )
    _release_accelerator_cache()
