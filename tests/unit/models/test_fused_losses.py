"""Fused Triton losses vs their eager references (speculators.losses.fused).

One test per loss, comparing loss value and logits-gradient against eager in
the three regimes that catch distinct bugs: fp32 with saturated point-mass
rows (gradient formula, log-space underflow), bf16 (the training dtype), and
the 151936-wide vocab (multi-block streaming, non-power-of-2 tail). The
upstream gradient contains exact zeros, so masked rows take the backward
kernel's early-out and must return exact zeros from an uninitialized buffer.

The accelerator is auto-detected: CUDA when present, otherwise Ascend NPU
(via triton-ascend). The 151936-wide leg also exercises the smaller NPU
BLOCK_SIZE cap (MAX_FUSED_SIZE_NPU = 4096), which forces the tighter
multi-block streaming loop.
"""

from types import SimpleNamespace

import pytest
import torch

from speculators.losses import eager, resolve_loss_config
from speculators.utils.util import is_npu_available


def _accelerator_device() -> str | None:
    """Pick the accelerator the fused kernels can run on, or None to skip."""
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    return None


DEVICE = _accelerator_device()
requires_accelerator = pytest.mark.skipif(
    DEVICE is None,
    reason="fused Triton losses require a CUDA or Ascend NPU accelerator",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="memory accounting test uses torch.cuda APIs",
)

# (name, eager fn, fused fn name); fused resolved lazily so this file
# collects on machines without Triton
CASES = [
    ("kl_div", eager.kl_div_loss, "fused_kl_div_loss"),
    ("rkl", eager.reverse_kl_div_loss, "fused_reverse_kl_div_loss"),
    ("jsd", eager.js_div_loss, "fused_js_div_loss"),
    ("ce", eager.ce_loss, "fused_ce_loss"),
    ("tv", eager.tv_loss, "fused_tv_loss"),
    ("nla", eager.neg_log_acceptance_loss, "fused_nla_loss"),
    ("lk_hybrid", eager.lk_hybrid_loss, "fused_lk_hybrid_loss"),
]

# Loss values are fp32 on both paths; gradients allow bf16 1-ulp rounding
# (both paths quantize at the leaf). A wrong gradient formula errs by orders
# of magnitude more than either bound.
LOSS_TOL = {"atol": 1e-4, "rtol": 1e-3}
GRAD_TOL = {"atol": 1e-3, "rtol": 1e-2}
# eager ce computes cross_entropy in bf16, so its bf16 legs are bounded by
# eager's own rounding, not by the (fp32) fused kernel
CE_BF16_LOSS_TOL = {"atol": 0.2, "rtol": 1e-2}
CE_BF16_GRAD_TOL = {"atol": 1e-2, "rtol": 2e-2}


def _assert_fused_matches_eager(
    eager_fn, fused_fn, logits, targets, loss_tol, grad_tol
):
    le = logits.detach().clone().requires_grad_(True)
    lf = logits.detach().clone().requires_grad_(True)
    out_e = eager_fn(le, targets)
    out_f = fused_fn(lf, targets)
    assert out_f.dtype == torch.float32  # like the eager fp32 softmax (#788)
    torch.testing.assert_close(out_f, out_e.float(), **loss_tol)

    go = torch.randn_like(out_e, dtype=torch.float32)
    go[:, ::3] = 0.0  # rows taking the go == 0 early-out
    (out_e.float() * go).sum().backward()
    (out_f * go).sum().backward()
    assert le.grad is not None
    assert lf.grad is not None
    torch.testing.assert_close(lf.grad.float(), le.grad.float(), **grad_tol)


@requires_accelerator
@pytest.mark.parametrize(
    ("name", "eager_fn", "fused_name"), CASES, ids=[c[0] for c in CASES]
)
def test_fused_matches_eager(name, eager_fn, fused_name):
    """Fused == eager (value + gradient) across the three failure-mode regimes."""
    fused_losses = pytest.importorskip("speculators.losses.fused")
    fused_fn = getattr(fused_losses, fused_name)

    # fp32, with saturated +-30 point-mass rows (one agreeing, one disagreeing)
    torch.manual_seed(0)
    logits = torch.randn(1, 32, 512, device=DEVICE) * 3
    targets = torch.randn(1, 32, 512, device=DEVICE) * 3
    logits[0, -2:] = -30.0
    targets[0, -2:] = -30.0
    logits[0, -2:, 0] = 30.0
    targets[0, -2, 0] = 30.0  # last two rows: draft==target, then disagree
    targets[0, -1, 7] = 30.0
    _assert_fused_matches_eager(eager_fn, fused_fn, logits, targets, LOSS_TOL, GRAD_TOL)
    # the map dispatcher is bitwise-equal to fused -> the fused path really ran
    dispatcher = resolve_loss_config(name)[name][0]
    assert torch.equal(dispatcher(logits, targets), fused_fn(logits, targets))

    # bf16, the training dtype
    torch.manual_seed(1)
    logits = (torch.randn(1, 64, 512, device=DEVICE) * 3).bfloat16()
    targets = (torch.randn(1, 64, 512, device=DEVICE) * 3).bfloat16()
    _assert_fused_matches_eager(
        eager_fn,
        fused_fn,
        logits,
        targets,
        CE_BF16_LOSS_TOL if name == "ce" else LOSS_TOL,
        CE_BF16_GRAD_TOL if name == "ce" else GRAD_TOL,
    )

    # Qwen3's 151936 vocab exceeds MAX_FUSED_SIZE (and MAX_FUSED_SIZE_NPU):
    # multi-block streaming plus a non-power-of-2 masked tail. On NPU the
    # per-block cap is 4096, so this leg also covers the tighter block loop.
    torch.manual_seed(2)
    logits = torch.randn(1, 3, 151936, device=DEVICE) * 3
    targets = torch.randn(1, 3, 151936, device=DEVICE) * 3
    _assert_fused_matches_eager(eager_fn, fused_fn, logits, targets, LOSS_TOL, GRAD_TOL)


@requires_accelerator
@pytest.mark.parametrize(
    ("name", "eager_fn", "fused_name"), CASES, ids=[c[0] for c in CASES]
)
def test_compiles_fullgraph(name, eager_fn, fused_name):
    """torch.compile(fullgraph=True) must trace the fused losses.

    The OP selector crosses the autograd.Function boundary, and Dynamo cannot
    represent a tl.constexpr object there -- passing one graph-breaks (or
    fails under fullgraph). Model forwards are wrapped in torch.compile, so
    this guards the compiled training path.
    """
    fused_losses = pytest.importorskip("speculators.losses.fused")
    fused_fn = getattr(fused_losses, fused_name)
    logits = torch.randn(1, 8, 512, device=DEVICE, requires_grad=True)
    targets = torch.randn(1, 8, 512, device=DEVICE)

    compiled = torch.compile(lambda a, b: fused_fn(a, b).sum(), fullgraph=True)
    compiled(logits, targets).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@requires_cuda
def test_ce_releases_targets_before_backward():
    """CE releases targets after forward; distribution losses retain them."""
    fused_losses = pytest.importorskip("speculators.losses.fused")

    def held_vs_target_bytes(fused_fn) -> tuple[int, int]:
        """(bytes the graph still holds after the forward, size of targets)."""
        logits = torch.randn(1, 1024, 4096, device="cuda", requires_grad=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        targets = torch.randn_like(logits)
        loss = fused_fn(logits, targets).sum()
        nbytes = targets.nbytes
        del targets
        held = torch.cuda.memory_allocated() - base
        loss.backward()  # backward must still run without the targets
        return held, nbytes

    held, nbytes = held_vs_target_bytes(fused_losses.fused_ce_loss)
    assert held < nbytes // 2
    held, nbytes = held_vs_target_bytes(fused_losses.fused_kl_div_loss)
    assert held >= nbytes


def test_eager_implementation_supports_differentiable_targets():
    """The explicit eager implementation preserves target gradients."""
    torch.manual_seed(3)
    for name in ("kl_div", "rkl", "jsd", "tv", "nla", "lk_hybrid"):
        logits = torch.randn(1, 4, 64, requires_grad=True)
        targets = torch.randn(1, 4, 64, requires_grad=True)
        loss_fn = resolve_loss_config(name, "eager")[name][0]
        loss_fn(logits, targets).sum().backward()
        assert logits.grad is not None, name
        assert targets.grad is not None, name


def test_calculate_settings_respects_device_cap():
    """`_calculate_settings` picks the right BLOCK_SIZE for each device without
    needing NPU or CUDA hardware -- the helper only reads ``device.type``.

    Exercises the NPU cap (MAX_FUSED_SIZE_NPU = 4096) and the CUDA cap
    (MAX_FUSED_SIZE = 131072) at vocab sizes that span the boundaries, so
    upstream CI (which typically has no NPU) still covers the NPU branch.
    """
    fused_losses = pytest.importorskip("speculators.losses.fused")

    npu_cases = (
        (512, 512),
        (4096, 4096),
        (8192, 4096),
        (32768, 4096),
        (131072, 4096),
        (151936, 4096),
    )
    for vocab, expected in npu_cases:
        block, _ = fused_losses._calculate_settings(vocab, SimpleNamespace(type="npu"))
        assert block == expected, (
            f"NPU cap: vocab={vocab} -> BLOCK_SIZE={block}, expected {expected}"
        )

    cuda_cases = (
        (512, 512),
        (4096, 4096),
        (8192, 8192),
        (131072, 131072),
        (151936, 131072),
    )
    for vocab, expected in cuda_cases:
        block, _ = fused_losses._calculate_settings(vocab, SimpleNamespace(type="cuda"))
        assert block == expected, (
            f"CUDA cap: vocab={vocab} -> BLOCK_SIZE={block}, expected {expected}"
        )
