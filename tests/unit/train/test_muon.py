from typing import Any

import pytest
import torch
from torch.optim._muon import _zeropower_via_newtonschulz

from speculators.train import muon
from speculators.train.muon import Muon, zeropower_via_newtonschulz

COEFFICIENTS = (3.4445, -4.7750, 2.0315)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="symmetric Gram GEMMs + Triton need CUDA"
)


def _grad(rows: int, cols: int, device: str) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(0)
    # Low-rank-dominated, like a real gradient.
    noise = torch.randn(rows, cols, generator=gen, device=device) * 0.01
    u = torch.randn(rows, 1, generator=gen, device=device)
    v = torch.randn(1, cols, generator=gen, device=device)
    return noise + 0.05 * u * v


@requires_cuda
@pytest.mark.parametrize("shape", [(4096, 8192), (8192, 4096), (4096, 4096)])
def test_matches_reference_on_cuda(shape):
    rows, cols = shape
    assert muon._use_symmetric(min(shape)), "shape no longer takes the split path"
    grad = _grad(rows, cols, device="cuda")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("shape", [(256, 512), (512, 256)])
def test_cpu_mirror_matches_reference(shape, monkeypatch):
    monkeypatch.setattr(muon, "_MIN_SYMMETRIC_ROWS", 256)
    rows, cols = shape
    grad = _grad(rows, cols, device="cpu")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    torch.testing.assert_close(ours, ref, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_symmetric_gram_is_symmetric_and_exact(device):
    x = torch.randn(512, 768, device=device, dtype=torch.bfloat16)
    out = torch.empty(512, 512, device=device, dtype=torch.bfloat16)

    muon._symmetric_gram(x, out, 4)

    torch.testing.assert_close(out, out.T, atol=0, rtol=0)
    torch.testing.assert_close(out.float(), (x @ x.T).float(), atol=0.5, rtol=1e-2)


def test_small_matrices_use_the_reference():
    grad = _grad(64, 128, device="cpu")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    torch.testing.assert_close(ours, ref, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1024, 2048, 2560, 3072])
def test_row_counts_the_split_loses_on_stay_on_the_reference(rows):
    # The split only beats a dense GEMM from 4096 rows up; at 2048 it costs 2.4x,
    # which is exactly the shape a Qwen3-1.7B draft is built from.
    assert not muon._use_symmetric(rows)


@pytest.mark.parametrize("rows", [4096, 5120, 8192])
def test_row_counts_the_split_wins_on_take_it(rows):
    assert muon._use_symmetric(rows)


@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_step_matches_torch_muon(adjust_lr_fn):
    torch.manual_seed(0)
    weights = [torch.randn(32, 64), torch.randn(48, 16)]
    ours = [torch.nn.Parameter(w.clone()) for w in weights]
    ref = [torch.nn.Parameter(w.clone()) for w in weights]
    kwargs: dict[str, Any] = {
        "lr": 2e-2,
        "weight_decay": 0.1,
        "adjust_lr_fn": adjust_lr_fn,
    }
    opt = Muon(ours, **kwargs)
    ref_opt = torch.optim.Muon(ref, **kwargs)

    for _ in range(3):
        grads = [torch.randn_like(w) for w in weights]
        for p, r, g in zip(ours, ref, grads, strict=True):
            p.grad, r.grad = g.clone(), g.clone()
        opt.step()
        ref_opt.step()

    for p, r in zip(ours, ref, strict=True):
        torch.testing.assert_close(p, r, atol=0, rtol=0)
        torch.testing.assert_close(
            opt.state[p]["momentum_buffer"], ref_opt.state[r]["momentum_buffer"]
        )


@requires_cuda
@pytest.mark.parametrize("shape", [(4096, 8192), (8192, 4096)])
def test_step_matches_torch_muon_on_the_symmetric_path(shape):
    torch.manual_seed(0)
    rows, cols = shape
    assert muon._use_symmetric(min(shape)), "shape no longer takes the split path"
    weight = torch.randn(rows, cols, device="cuda") * 0.02
    ours = torch.nn.Parameter(weight.clone())
    ref = torch.nn.Parameter(weight.clone())
    kwargs: dict[str, Any] = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "adjust_lr_fn": "match_rms_adamw",
    }
    opt = Muon([ours], **kwargs)
    ref_opt = torch.optim.Muon([ref], **kwargs)

    for _ in range(3):
        grad = _grad(rows, cols, device="cuda") + torch.randn_like(weight) * 1e-3
        ours.grad, ref.grad = grad.clone(), grad.clone()
        opt.step()
        ref_opt.step()

    torch.testing.assert_close(
        opt.state[ours]["momentum_buffer"],
        ref_opt.state[ref]["momentum_buffer"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)
