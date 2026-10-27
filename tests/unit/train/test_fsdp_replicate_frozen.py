"""Test frozen-parameter selection and FSDP wiring without a process group."""

import pytest
import torch
from torch import nn

from speculators.train import distributed as dist_mod
from speculators.train.distributed import _frozen_params, apply_fully_sharded


class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.frozen_head = nn.Linear(4, 8).requires_grad_(False)


def test_frozen_params_selects_only_frozen():
    model = MixedModel()
    assert _frozen_params(model) == set(model.frozen_head.parameters())


@pytest.mark.parametrize("frozen", [False, True])
def test_frozen_params_uniform_model(frozen):
    model = nn.Linear(4, 4).requires_grad_(not frozen)
    assert _frozen_params(model) == (set(model.parameters()) if frozen else set())


@pytest.mark.parametrize("replicate", [False, True])
def test_apply_fully_sharded_ignored_params(monkeypatch, replicate):
    model = MixedModel()
    frozen = set(model.frozen_head.parameters())
    calls = []

    def record_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(dist_mod, "fully_shard", record_shard)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    assert apply_fully_sharded(model, replicate_frozen=replicate) is model
    assert [module for module, _ in calls] == [model.layers[0], model]
    if replicate:
        assert calls[-1][1]["ignored_params"] == frozen
    else:
        assert all("ignored_params" not in kwargs for _, kwargs in calls)


def test_apply_fully_sharded_without_frozen_params(monkeypatch):
    model = MixedModel().requires_grad_(True)
    calls = []
    monkeypatch.setattr(
        dist_mod, "fully_shard", lambda module, **kwargs: calls.append(kwargs)
    )
    apply_fully_sharded(model, replicate_frozen=True)
    assert all("ignored_params" not in kwargs for kwargs in calls)


def test_ignored_params_moved_and_cast_after_wrapping(monkeypatch):
    model = MixedModel()
    frozen = set(model.frozen_head.parameters())
    trainable = list(model.layers.parameters())
    original = {param: param.detach().clone() for param in frozen}

    def record_shard(module, **kwargs):
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
