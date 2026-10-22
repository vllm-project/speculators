from types import SimpleNamespace

import pytest
import torch
from torch.optim._muon import _zeropower_via_newtonschulz

import speculators.train.optimizers as optimizers_module
from speculators.train.config import TrainConfig
from speculators.train.optimizers import (
    Muon,
    _newtonschulz,
    _symmetric_gram,
    build_optimizers,
)


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


def test_muon_builds_the_symmetric_gram_subclass():
    optimizers = build_optimizers(_model(), _config("muon"))

    muon = [opt for opt in optimizers if isinstance(opt, torch.optim.Muon)]
    assert len(muon) == 1
    assert isinstance(muon[0], Muon)
    # Logged learning rates are keyed by the optimizer's class name.
    assert type(muon[0]).__name__ == "Muon"


@pytest.mark.parametrize("num_blocks", [2, 4])
def test_symmetric_gram_matches_full_matmul(num_blocks):
    x = torch.randn(64, 256, dtype=torch.float64)
    out = torch.empty(64, 64, dtype=torch.float64)

    _symmetric_gram(x, out, num_blocks)

    torch.testing.assert_close(out, x @ x.T)
    assert torch.equal(out, out.T)


@pytest.mark.parametrize("shape", [(64, 16), (16, 64), (32, 32), (16, 6)])
def test_newtonschulz_matches_torch(monkeypatch, shape):
    """The block Gram path (and its fallbacks) reproduce torch's iteration."""
    monkeypatch.setattr(optimizers_module, "_SYMMETRIC_GRAM_MIN_BLOCK", 4)
    grad = torch.randn(*shape)
    coefficients = (3.4445, -4.7750, 2.0315)

    expected = _zeropower_via_newtonschulz(grad, coefficients, 5, 1e-7)
    normalized = grad.bfloat16()
    normalized.div_(normalized.norm())
    actual = _newtonschulz(normalized, coefficients, 5)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=0)


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_step_matches_torch_muon(monkeypatch, nesterov):
    """Same params, grads and hyperparameters give the torch.optim.Muon update,
    including across a learning-rate change."""
    monkeypatch.setattr(optimizers_module, "_SYMMETRIC_GRAM_MIN_BLOCK", 4)
    torch.manual_seed(0)
    shapes = [(64, 16), (16, 64), (32, 32)]
    ours = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
    ref = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    kwargs = {
        "lr": 2e-2,
        "momentum": 0.95,
        "weight_decay": 0.01,
        "ns_steps": 5,
        "nesterov": nesterov,
    }
    opt_ours, opt_ref = Muon(ours, **kwargs), torch.optim.Muon(ref, **kwargs)

    for step in range(3):
        for opt in (opt_ours, opt_ref):
            opt.param_groups[0]["lr"] = 2e-2 / (step + 1)
        for p_ours, p_ref in zip(ours, ref, strict=True):
            p_ours.grad = torch.randn_like(p_ours)
            p_ref.grad = p_ours.grad.clone()
        opt_ours.step()
        opt_ref.step()

    for p_ours, p_ref in zip(ours, ref, strict=True):
        torch.testing.assert_close(p_ours, p_ref, atol=1e-4, rtol=0)
    for p_ours, p_ref in zip(ours, ref, strict=True):
        torch.testing.assert_close(
            opt_ours.state[p_ours]["momentum_buffer"],
            opt_ref.state[p_ref]["momentum_buffer"],
        )
