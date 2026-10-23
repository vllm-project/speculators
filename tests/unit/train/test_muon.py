import builtins
import importlib.util
from typing import Any

import pytest
import torch
from torch.optim._muon import _zeropower_via_newtonschulz

from speculators.train import muon
from speculators.train.muon import Muon, zeropower_via_newtonschulz

COEFFICIENTS = (3.4445, -4.7750, 2.0315)
# Tolerance for "the split computes the same thing as a dense GEMM": the two
# differ only in which cuBLAS kernel runs, so this is bf16 GEMM noise.
SPLIT_ATOL = 2e-3
SPLIT_RTOL = 2e-2

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


@pytest.fixture
def fallbacks(monkeypatch) -> list[tuple[int, ...]]:
    """Shapes that fell back to upstream, in call order.

    ``zeropower_via_newtonschulz`` silently defers to
    ``_zeropower_via_newtonschulz`` for DTensors, for shapes the split does not
    pay on, and without Triton, so a test of the symmetric path only tests it
    if nothing fell back -- and a threshold change could flip that without
    failing anything. Asserting on the calls is the direct check;
    ``_use_symmetric`` is only a proxy for it.
    """
    calls: list[tuple[int, ...]] = []
    reference = muon._zeropower_via_newtonschulz

    def counting(grad, *args, **kwargs):
        calls.append(tuple(grad.shape))
        return reference(grad, *args, **kwargs)

    monkeypatch.setattr(muon, "_zeropower_via_newtonschulz", counting)
    return calls


@pytest.fixture
def fused(monkeypatch) -> list[tuple[int, ...]]:
    """Shapes that took the fused elementwise path, in call order.

    The fused and eager paths are bitwise equal, so an equivalence test passes
    either way; ``_fusable`` turning itself off would go unnoticed.
    """
    calls: list[tuple[int, ...]] = []
    reference = muon._momentum_update

    def counting(grad, *args, **kwargs):
        calls.append(tuple(grad.shape))
        return reference(grad, *args, **kwargs)

    monkeypatch.setattr(muon, "_momentum_update", counting)
    return calls


@requires_cuda
@pytest.mark.parametrize("shape", [(4096, 8192), (8192, 4096), (4096, 4096)])
def test_split_matches_the_reference_on_cuda(shape, fallbacks):
    rows, cols = shape
    grad = _grad(rows, cols, device="cuda")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    assert not fallbacks, f"{shape} took the reference path, so this compares nothing"
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, atol=SPLIT_ATOL, rtol=SPLIT_RTOL)


@pytest.mark.parametrize("shape", [(256, 512), (512, 256)])
def test_split_matches_the_reference_on_cpu(shape, monkeypatch, fallbacks):
    # CPU mirrors with the block-copy loop rather than the Triton kernel. 256 is
    # the smallest tile-aligned size; the real threshold is a CUDA crossover.
    monkeypatch.setattr(muon, "_MIN_SYMMETRIC_ROWS", 256)
    rows, cols = shape
    grad = _grad(rows, cols, device="cpu")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    assert not fallbacks, f"{shape} took the reference path, so this compares nothing"
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, atol=SPLIT_ATOL, rtol=SPLIT_RTOL)


@requires_cuda
@pytest.mark.parametrize("shape", [(4096, 8192), (8192, 4096)])
def test_cuda_matches_the_reference_without_triton(shape, monkeypatch, fallbacks):
    """No Triton means no mirror kernel, so CUDA falls back and is bit-exact.

    The block-copy loop would also be correct here, but the crossover was
    measured against the kernel and the loop does not clear it at every size
    past it -- so the fallback is the gate, not the loop.
    """
    monkeypatch.setattr(muon, "HAS_TRITON", False)
    rows, cols = shape
    grad = _grad(rows, cols, device="cuda")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    assert fallbacks == [(rows, cols)]
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)


def test_module_imports_without_triton(monkeypatch):
    """Triton must stay an optional dependency of this module.

    ``optimizers.py`` imports ``Muon`` at module scope whichever optimizer is
    configured, so a top-level ``import triton`` here would make an AdamW-only
    run fail at import on a build without it.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    spec = importlib.util.spec_from_file_location("muon_without_triton", muon.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not module.HAS_TRITON
    assert not hasattr(module, "_mirror_lower_blocks_kernel")
    assert not hasattr(module, "_nesterov_momentum_kernel")
    assert not hasattr(module, "_apply_update_kernel")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_symmetric_gram_is_symmetric_and_exact(device):
    x = torch.randn(512, 768, device=device, dtype=torch.bfloat16)
    out = torch.empty(512, 512, device=device, dtype=torch.bfloat16)

    muon._symmetric_gram(x, out, 4)

    torch.testing.assert_close(out, out.T, atol=0, rtol=0)
    torch.testing.assert_close(out.float(), (x @ x.T).float(), atol=0.5, rtol=1e-2)


def test_small_matrices_match_the_reference(fallbacks):
    """Below the crossover the fallback fires, and is upstream bit for bit."""
    grad = _grad(64, 128, device="cpu")

    ours = zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)
    ref = _zeropower_via_newtonschulz(grad, COEFFICIENTS, 5, 1e-7)

    assert fallbacks == [(64, 128)]
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1024, 2048, 2560, 3072])
def test_row_counts_the_split_loses_on_stay_on_the_reference(rows):
    # The split loses below 4096 rows; 2048-row matrices are what a Qwen3-1.7B
    # draft is built from.
    assert not muon._use_symmetric(rows)


@pytest.mark.parametrize("rows", [4096, 5120, 8192])
def test_row_counts_the_split_wins_on_take_it(rows):
    assert muon._use_symmetric(rows)


@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_step_matches_torch_muon(adjust_lr_fn, fallbacks):
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

    # These weights are far below the crossover, so every orthogonalization is
    # upstream's: 3 steps x 2 params. This is the half of the equivalence claim
    # that test_step_matches_torch_muon_on_the_symmetric_path does not cover.
    assert len(fallbacks) == 6

    for p, r in zip(ours, ref, strict=True):
        torch.testing.assert_close(p, r, atol=0, rtol=0)
        torch.testing.assert_close(
            opt.state[p]["momentum_buffer"], ref_opt.state[r]["momentum_buffer"]
        )


@requires_cuda
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("shape", [(2048, 4096), (4096, 2048), (1024, 512), (37, 100)])
def test_fused_step_is_bit_identical_to_torch_muon_on_cuda(
    shape, nesterov, monkeypatch, fused
):
    # Drop the size bound so the small and ragged shapes still reach the kernels.
    monkeypatch.setattr(muon, "_MIN_FUSED_NUMEL", 0)
    torch.manual_seed(0)
    rows, cols = shape
    weight = torch.randn(rows, cols, device="cuda") * 0.02
    ours = torch.nn.Parameter(weight.clone())
    ref = torch.nn.Parameter(weight.clone())
    kwargs: dict[str, Any] = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "nesterov": nesterov,
        "adjust_lr_fn": "match_rms_adamw",
    }
    opt = Muon([ours], **kwargs)
    ref_opt = torch.optim.Muon([ref], **kwargs)

    for _ in range(3):
        grad = _grad(rows, cols, device="cuda") + torch.randn_like(weight) * 1e-3
        ours.grad, ref.grad = grad.clone(), grad.clone()
        opt.step()
        ref_opt.step()

    assert fused == [(rows, cols)] * 3
    torch.testing.assert_close(
        opt.state[ours]["momentum_buffer"],
        ref_opt.state[ref]["momentum_buffer"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize(
    ("shape", "expected"), [((2048, 4096), True), ((1024, 512), False)]
)
def test_the_size_bound_picks_the_path(shape, expected):
    param = torch.zeros(*shape, device="cuda")
    assert muon._fusable(param, param, param) is expected


@requires_cuda
@pytest.mark.parametrize("shape", [(4096, 8192), (8192, 4096)])
def test_step_matches_torch_muon_on_the_symmetric_path(shape, fallbacks):
    torch.manual_seed(0)
    rows, cols = shape
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

    assert not fallbacks, f"{shape} took the reference path, so this compares nothing"
    torch.testing.assert_close(
        opt.state[ours]["momentum_buffer"],
        ref_opt.state[ref]["momentum_buffer"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)
