"""Deferred gradients, skipped rows and logit-tile dtype in fused_linear.

Deferred mode computes each differentiated term's unit-weight
``d loss / d hidden`` (``jac``) in forward and finishes backward as a per-row
scale and sum. It must equal the recompute mode exactly in fp64 and to
rounding on GPU, including D-PACE, whose loss-dependent weights are what rule
out finishing the gradient in forward.
"""

from functools import partial

import pytest
import torch

from speculators.losses import compound_loss_from_hidden, fused, resolve_loss_config
from speculators.losses import fused_linear as fl
from speculators.losses.utils import dpace_loss_decay
from speculators.models.dflash.metrics import compute_metrics_from_hidden
from tests.unit.models.test_fused_linear import _run
from tests.unit.models.test_fused_linear_targets import _inputs
from tests.unit.models.test_fused_losses import (
    DEVICE,
    requires_accelerator,
    requires_cuda,
)

pytestmark = pytest.mark.usefixtures("narrowest_tile")

NAME_SETS = [
    ("kl_div",),
    ("ce",),
    ("tv",),
    ("rkl",),
    ("jsd",),
    ("kl_div", "tv"),
    ("ce", "tv"),
    ("lk_hybrid",),
    ("nla",),
]
IDS = ["+".join(n) for n in NAME_SETS]


def _both_modes(h, w, t, names, target_weight=None, **kwargs):
    out = {}
    for mode in ("deferred", "recompute"):
        out[mode] = _run(
            lambda x, mode=mode: fl.fused_linear_compound(
                x,
                w,
                t,
                names,
                metric_terms=("tv",),
                target_weight=target_weight,
                grad_mode=mode,
                **kwargs,
            ),
            h,
            names,
        )
    return out["deferred"], out["recompute"]


# ---------------------------------------------------------------------------
# CPU (torch reference path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("names", NAME_SETS, ids=IDS)
@pytest.mark.parametrize("recompute_targets", [False, True])
def test_deferred_equals_recompute_fp64(names, recompute_targets):
    h, w, vh, tw = _inputs(13, 6, 5, 17, torch.float64, "cpu", seed=1)
    targets, target_weight = (vh, tw) if recompute_targets else (vh @ tw.T, None)
    (out, grad), (ref, ref_grad) = _both_modes(
        h, w, targets, names, target_weight, chunk_rows=5
    )
    for name in names:
        torch.testing.assert_close(out.losses[name], ref.losses[name])
    torch.testing.assert_close(grad, ref_grad)


@pytest.mark.parametrize("name", ["kl_div", "rkl", "jsd", "ce", "tv", "nla"])
def test_gradcheck_deferred(name):
    h, w, vh, tw = _inputs(13, 6, 5, 17, torch.float64, "cpu", seed=2)
    h.requires_grad_(True)

    def loss(x):
        out = fl.fused_linear_compound(
            x, w, vh, (name,), chunk_rows=5, target_weight=tw, grad_mode="deferred"
        )
        return out.losses[name]

    assert torch.autograd.gradcheck(loss, (h,))


@pytest.mark.parametrize("spec", ["ce", '{"kl_div": 0.5, "tv": 0.5}'])
def test_dpace_gradients_deferred_equal_recompute(spec):
    """D-PACE weights depend on the loss value; deferred mode stays exact
    because they arrive as per-row grad_output in backward."""
    h, w, vh, tw = _inputs(24, 6, 5, 17, torch.float64, "cpu", seed=3)
    loss_mask = torch.ones(1, 24, dtype=torch.long)
    loss_mask[0, 5:9] = 0
    pos_idx = (torch.arange(24) % 8).unsqueeze(0)
    decay_fn = partial(
        dpace_loss_decay, loss_mask=loss_mask, block_size=8, dpace_alpha=0.5
    )
    config = resolve_loss_config(spec)
    grads = {}
    for mode in ("deferred", "recompute"):
        x = h.clone().requires_grad_(True)
        out = compound_loss_from_hidden(
            x[None],
            w,
            vh[None],
            loss_mask,
            pos_idx,
            config,
            decay_fn=decay_fn,
            chunk_rows=7,
            target_weight=tw,
            grad_mode=mode,
        )
        out.loss.backward()
        grads[mode] = (out.loss.detach(), x.grad)
    torch.testing.assert_close(grads["deferred"][0], grads["recompute"][0])
    torch.testing.assert_close(grads["deferred"][1], grads["recompute"][1])


@pytest.mark.parametrize("grad_mode", ["deferred", "recompute"])
def test_skipped_rows_are_fully_masked(grad_mode):
    """``rows`` computes only the listed rows; the rest report 0 and get no
    gradient, and computed rows equal the full computation."""
    h, w, vh, tw = _inputs(16, 6, 5, 17, torch.float64, "cpu", seed=4)
    names = ("kl_div", "ce")
    keep = torch.tensor([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
    skipped = torch.tensor([0, 4, 8, 12])
    common = {"target_weight": tw, "grad_mode": grad_mode, "chunk_rows": 5}
    out, grad = _run(
        lambda x: fl.fused_linear_compound(x, w, vh, names, rows=keep, **common),
        h,
        names,
    )
    full, full_grad = _run(
        lambda x: fl.fused_linear_compound(x, w, vh, names, **common), h, names
    )
    for name in names:
        assert torch.count_nonzero(out.losses[name][skipped]) == 0
        torch.testing.assert_close(out.losses[name][keep], full.losses[name][keep])
    assert torch.count_nonzero(out.pred_ids[skipped]) == 0
    assert torch.equal(out.pred_ids[keep], full.pred_ids[keep])
    assert torch.count_nonzero(grad[skipped]) == 0
    # _run weights rows by (row % 3); the kept rows' gradients are unchanged
    torch.testing.assert_close(grad[keep], full_grad[keep])


def test_no_grad_computes_losses_only(monkeypatch):
    h, w, vh, tw = _inputs(10, 6, 5, 17, torch.float64, "cpu", seed=5)
    calls: list[int] = []
    real = fl._forward_deferred_chunk

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(fl, "_forward_deferred_chunk", spy)
    with torch.no_grad():
        out = fl.fused_linear_compound(
            h.requires_grad_(True), w, vh, ("kl_div",), target_weight=tw
        )
    assert not calls
    assert out.losses["kl_div"].grad_fn is None
    fl.fused_linear_compound(h, w, vh, ("kl_div",), target_weight=tw)
    assert calls


def test_option_validation():
    h, w, vh, tw = _inputs(8, 4, 3, 10, torch.float32, "cpu")
    with pytest.raises(ValueError, match="grad_mode"):
        fl.fused_linear_compound(h, w, vh, ("kl_div",), target_weight=tw, grad_mode="x")
    with pytest.raises(ValueError, match="logits_dtype"):
        fl.fused_linear_compound(
            h, w, vh, ("kl_div",), target_weight=tw, logits_dtype="bf16"
        )


@pytest.mark.parametrize(
    ("rows", "expected"), [(15360, 2048), (16384, 2048), (1500, 1500), (100, 100)]
)
def test_chunk_rows_capped_at_2048(rows, expected, monkeypatch):
    """2048-row chunks (remainder tail): the fastest measured shape for both
    the projection and the dz @ W contraction on MI355X."""
    monkeypatch.delenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", raising=False)
    assert fl.chunk_rows_for(151936, torch.bfloat16, rows, recompute_targets=True) == (
        expected
    )


# ---------------------------------------------------------------------------
# GPU (Triton loss_grad_kernel)
# ---------------------------------------------------------------------------


def _kernel_reference(z, t, names):
    """Losses and per-term unit-weight gradients from the existing kernels."""
    rows, vocab = z.shape
    losses = torch.zeros(5, rows, device=z.device)
    argmax = torch.empty(2, rows, device=z.device, dtype=torch.int64)
    stats = torch.empty(8, rows, device=z.device)
    fused.loss_forward_kernel[(rows,)](
        z,
        t,
        losses,
        rows,
        argmax,
        rows,
        stats,
        rows,
        vocab,
        TERMS=fused.terms_mask(names),
    )
    grads = {}
    for name in names:
        go = torch.zeros(5, rows, device=z.device)
        go[fused.TERM_SLOTS[name]] = 1.0
        grad = torch.empty_like(z)
        fused.loss_backward_kernel[(rows,)](
            z, t, grad, go, rows, stats, rows, vocab, TERMS=fused.terms_mask((name,))
        )
        grads[name] = grad
    return losses, grads


@requires_accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("vocab", [1000, 151936])
@pytest.mark.parametrize(
    "names",
    [
        ("kl_div",),
        ("ce",),
        ("kl_div", "tv"),
        ("rkl", "jsd"),
        ("kl_div", "rkl", "jsd", "ce", "tv"),
    ],
    ids="+".join,
)
def test_loss_grad_kernel_matches_existing_kernels(dtype, vocab, names):
    torch.manual_seed(0)
    rows = 64
    z = (torch.randn(rows, vocab, device=DEVICE) * 3).to(dtype)
    t = (torch.randn(rows, vocab, device=DEVICE) * 3).to(dtype)
    ties = torch.randint(-3, 4, (rows, vocab), device=DEVICE).to(dtype)
    for logits, targets in ((z, t), (ties, ties.flip(0))):
        losses = torch.zeros(5, rows, device=DEVICE)
        argmax = torch.empty(2, rows, device=DEVICE, dtype=torch.int64)
        dz = torch.empty(len(names), rows, vocab, device=DEVICE, dtype=dtype)
        slot = {n: i for i, n in enumerate(names)}
        fused.loss_grad_kernel[(rows,)](
            logits,
            targets,
            losses,
            rows,
            argmax,
            rows,
            dz,
            rows * vocab,
            vocab,
            TERMS=fused.terms_mask(names),
            GRAD_TERMS=fused.terms_mask(names),
            DZ_KL=slot.get("kl_div", 0),
            DZ_RKL=slot.get("rkl", 0),
            DZ_JSD=slot.get("jsd", 0),
            DZ_CE=slot.get("ce", 0),
            DZ_TV=slot.get("tv", 0),
        )
        assert torch.equal(argmax[0], logits.argmax(-1))
        assert torch.equal(argmax[1], targets.argmax(-1))
        ref_losses, ref_grads = _kernel_reference(logits, targets, names)
        for name in names:
            s = fused.TERM_SLOTS[name]
            torch.testing.assert_close(losses[s], ref_losses[s], rtol=1e-5, atol=1e-5)
            ref = ref_grads[name].float()
            torch.testing.assert_close(
                dz[slot[name]].float(),
                ref,
                rtol=1e-3,
                atol=1e-6 + 1e-3 * float(ref.abs().max()),
            )


@requires_accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("names", NAME_SETS, ids=IDS)
@pytest.mark.parametrize("recompute_targets", [False, True])
def test_deferred_matches_recompute_gpu(dtype, names, recompute_targets):
    h, w, vh, tw = _inputs(300, 64, 48, 3000, dtype, DEVICE, seed=6)
    targets, target_weight = (vh, tw) if recompute_targets else (vh @ tw.T, None)
    (out, grad), (ref, ref_grad) = _both_modes(
        h, w, targets, names, target_weight, chunk_rows=128
    )
    for name in names:
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-5, atol=1e-5
        )
    assert torch.equal(out.pred_ids, ref.pred_ids)
    # recompute rounds g * dz to the input dtype, deferred rounds dz then scales
    scale = 1e-4 if dtype == torch.float32 else 2e-2
    atol = scale * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=scale * 10, atol=atol)


@requires_accelerator
def test_dflash_dpace_metrics_deferred_equal_recompute():
    h, w, vh, tw = _inputs(64, 32, 24, 1000, torch.float32, DEVICE, seed=7)
    loss_mask = torch.ones(1, 64, device=DEVICE, dtype=torch.long)
    loss_mask[0, ::8] = 0  # DFlash masks the anchor slot
    config = resolve_loss_config('{"kl_div": 0.5, "tv": 0.5}')
    results = {}
    for mode in ("deferred", "recompute"):
        x = h.clone().requires_grad_(True)
        loss, metrics = compute_metrics_from_hidden(
            x[None],
            w,
            vh[None],
            loss_mask,
            block_size=8,
            loss_config=config,
            per_position_loss_weight="dpace",
            target_weight=tw,
            grad_mode=mode,
            skip_anchor_rows=True,
            chunk_rows=24,
        )
        loss.backward()
        results[mode] = (loss.detach(), metrics, x.grad)
    d, r = results["deferred"], results["recompute"]
    torch.testing.assert_close(d[0], r[0])
    for key, value in r[1].items():
        torch.testing.assert_close(d[1][key], value, msg=key)
    torch.testing.assert_close(d[2], r[2], rtol=1e-4, atol=1e-7)
    assert torch.count_nonzero(d[2][::8]) == 0  # skipped anchor rows


@requires_cuda
def test_deferred_keeps_only_jac_for_backward():
    h, w, vh, tw = _inputs(4096, 256, 256, 16000, torch.bfloat16, "cuda", seed=8)
    x = h.clone().requires_grad_(True)
    names = ("kl_div",)
    fl.fused_linear_compound(x, w, vh, names, target_weight=tw)  # autotune
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    out = fl.fused_linear_compound(x, w, vh, names, target_weight=tw)
    loss = out.losses["kl_div"].sum()
    del out
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before
    jac_bytes = 4096 * 256 * 4  # fp32: rounded once, in backward
    per_row_outputs = 4096 * (5 * 4 + 2 * 8 + 8 * 4)
    assert held <= jac_bytes + per_row_outputs + 1024**2
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
