"""Test frozen-parameter selection and FSDP wiring without a process group."""

import pytest
import torch
from torch import nn

from speculators.train import distributed as dist_mod
from speculators.train.distributed import _frozen_params, apply_fully_sharded


class MixedModel(nn.Module):
    """Model with a trainable decoder layer and a frozen output head."""

    def __init__(self):
        """Build a small model for CPU-only FSDP wiring tests."""
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.frozen_head = nn.Linear(4, 8).requires_grad_(False)


def test_frozen_params_selects_only_frozen():
    """Select frozen head parameters while excluding trainable layer weights."""
    model = MixedModel()
    assert _frozen_params(model) == set(model.frozen_head.parameters())


@pytest.mark.parametrize("frozen", [False, True])
def test_frozen_params_uniform_model(frozen):
    """Handle models whose parameters are all trainable or all frozen."""
    model = nn.Linear(4, 4).requires_grad_(not frozen)
    assert _frozen_params(model) == (set(model.parameters()) if frozen else set())


@pytest.mark.parametrize("replicate", [False, True])
def test_apply_fully_sharded_ignored_params(monkeypatch, replicate):
    """Exclude frozen head and decoder-layer parameters only when enabled."""
    model = MixedModel()
    layer_bias = model.layers[0].bias.requires_grad_(False)
    frozen = set(model.frozen_head.parameters()) | {layer_bias}
    calls = []

    def record_shard(module, **kwargs):
        """Capture layer and root wrapping arguments without initializing FSDP."""
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(dist_mod, "fully_shard", record_shard)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    assert apply_fully_sharded(model, replicate_frozen=replicate) is model
    assert [module for module, _ in calls] == [model.layers[0], model]
    if replicate:
        assert calls[0][1]["ignored_params"] == frozen
        assert calls[-1][1]["ignored_params"] == frozen
    else:
        assert all("ignored_params" not in kwargs for _, kwargs in calls)


def test_apply_fully_sharded_without_frozen_params(monkeypatch):
    """Keep standard wrapping when replication has no frozen weights to exclude."""
    model = MixedModel().requires_grad_(True)
    calls = []
    monkeypatch.setattr(
        dist_mod, "fully_shard", lambda module, **kwargs: calls.append(kwargs)
    )
    apply_fully_sharded(model, replicate_frozen=True)
    assert all("ignored_params" not in kwargs for kwargs in calls)


def test_ignored_params_moved_and_cast_after_wrapping(monkeypatch):
    """Preserve frozen weights and identity when moving them after wrapping."""
    model = MixedModel()
    frozen = set(model.frozen_head.parameters())
    trainable = list(model.layers.parameters())
    original = {param: param.detach().clone() for param in frozen}

    def record_shard(module, **kwargs):
        """Verify weights retain their original dtype until wrapping completes."""
        # Ignored weights are converted only after the root has been wrapped.
        assert all(param.dtype == torch.float32 for param in frozen)
        return module

    monkeypatch.setattr(dist_mod, "fully_shard", record_shard)
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("cpu")
    )
    apply_fully_sharded(model, replicate_frozen=True, param_dtype=torch.bfloat16)
    assert set(model.frozen_head.parameters()) == frozen
    for param in frozen:
        assert param.device.type == "cpu"
        assert param.dtype == torch.bfloat16
        assert not param.requires_grad
        torch.testing.assert_close(param, original[param].to(torch.bfloat16))
    assert all(param.dtype == torch.float32 for param in trainable)
