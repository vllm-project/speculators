from types import SimpleNamespace

import pytest
import torch

from speculators.train.optimizers import build_optimizers


def _model() -> torch.nn.Module:
    """A model with both 2D matrices (Muon) and 1D norms/biases (AdamW)."""
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))


def _config(optimizer: str) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer=optimizer,
        lr=1e-3,
        weight_decay=0.01,
        muon_lr=2e-2,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        muon_ns_steps=5,
        muon_adjust_lr_fn=None,
    )


def test_adamw_is_fused():
    (optimizer,) = build_optimizers(_model(), _config("adamw"))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["fused"] for group in optimizer.param_groups)


def test_muon_adamw_group_is_fused():
    optimizers = build_optimizers(_model(), _config("muon"))

    adamw = [opt for opt in optimizers if isinstance(opt, torch.optim.AdamW)]
    assert adamw, "muon should pair Muon with AdamW for the non-matrix params"
    assert all(group["fused"] for opt in adamw for group in opt.param_groups)


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_step_runs(optimizer):
    """The fused path rejects unsupported devices/dtypes at ``step``, not init."""
    model = _model()
    optimizers = build_optimizers(model, _config(optimizer))
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    for opt in optimizers:
        opt.step()


def test_unsupported_optimizer_raises():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizers(_model(), _config("sgd"))
