from types import SimpleNamespace

import pytest
import torch

from speculators.train.config import TrainConfig
from speculators.train.optimizers import build_optimizers


def _model() -> torch.nn.Module:
    """A model with both 2D matrices (Muon) and 1D norms/biases (AdamW)."""
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))


def _config(optimizer: str, implementation: str = "fused") -> SimpleNamespace:
    return SimpleNamespace(
        optimizer=optimizer,
        optimizer_implementation=implementation,
        lr=1e-3,
        weight_decay=0.01,
        muon_lr=2e-2,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        muon_ns_steps=5,
        muon_adjust_lr_fn=None,
    )


@pytest.mark.parametrize(
    ("implementation", "fused"), [("fused", True), ("eager", False)]
)
def test_adamw_follows_implementation(implementation, fused):
    (optimizer,) = build_optimizers(_model(), _config("adamw", implementation))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["fused"] is fused for group in optimizer.param_groups)


@pytest.mark.parametrize(
    ("implementation", "fused"), [("fused", True), ("eager", False)]
)
def test_muon_adamw_group_follows_implementation(implementation, fused):
    optimizers = build_optimizers(_model(), _config("muon", implementation))

    adamw = [opt for opt in optimizers if isinstance(opt, torch.optim.AdamW)]
    assert adamw, "muon should pair Muon with AdamW for the non-matrix params"
    assert all(group["fused"] is fused for opt in adamw for group in opt.param_groups)


def test_fused_is_the_default():
    assert (
        TrainConfig.resolve(
            ["--verifier-name-or-path", "dummy"]
        ).optimizer.optimizer_implementation
        == "fused"
    )


def test_implementation_comes_from_the_cli():
    cfg = TrainConfig.resolve(
        ["--verifier-name-or-path", "dummy", "--optimizer-implementation", "eager"]
    )
    assert cfg.optimizer.optimizer_implementation == "eager"


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("implementation", ["fused", "eager"])
def test_step_runs(optimizer, implementation):
    """The fused path rejects unsupported devices/dtypes at ``step``, not init."""
    model = _model()
    optimizers = build_optimizers(model, _config(optimizer, implementation))
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    for opt in optimizers:
        opt.step()


def test_unsupported_optimizer_raises():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizers(_model(), _config("sgd"))
