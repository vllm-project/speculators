"""Tests for speculators.train.optimizers."""

import pytest
import torch

from speculators.train.optimizers import Muon

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bitwise bmm/mm parity is checked on CUDA"
)

# Two shape groups that batch (one via transposition) plus a singleton.
_SHAPES = [(256, 512), (256, 512), (512, 256), (384, 384), (96, 160)]


@requires_cuda
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_batched_muon_matches_torch_muon(nesterov, adjust_lr_fn):
    """Shape-batched Newton-Schulz is bitwise identical to torch.optim.Muon."""
    torch.manual_seed(0)
    ref_params = [
        torch.nn.Parameter(torch.randn(s, device="cuda") * 0.02) for s in _SHAPES
    ]
    params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    kwargs = {
        "lr": 0.02,
        "momentum": 0.95,
        "weight_decay": 0.01,
        "nesterov": nesterov,
        "adjust_lr_fn": adjust_lr_fn,
    }
    ref_opt = torch.optim.Muon(ref_params, **kwargs)
    opt = Muon(params, **kwargs)

    for _ in range(3):
        for ref, p in zip(ref_params, params, strict=True):
            grad = torch.randn_like(ref)
            ref.grad = grad.clone()
            p.grad = grad.clone()
        ref_opt.step()
        opt.step()

    for ref, p in zip(ref_params, params, strict=True):
        assert torch.equal(ref, p)
        assert torch.equal(
            ref_opt.state[ref]["momentum_buffer"], opt.state[p]["momentum_buffer"]
        )

    # Same state layout: checkpoints load across the two implementations.
    ref_opt.load_state_dict(opt.state_dict())
    assert type(opt).__name__ == "Muon"  # lr metrics are keyed by class name
