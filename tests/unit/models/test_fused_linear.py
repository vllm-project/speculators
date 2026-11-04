"""Fused linear losses (speculators.losses.fused_linear).

The op computes the fused losses from hidden states and the lm_head weight in
row chunks; the [N, V] draft logits and their gradient never exist. CPU runs a
torch reference of the per-chunk math (fp64 gradcheck, opcheck); GPU runs the
Triton kernels and is checked against the materialized fused path.
"""

from typing import Any

import pytest
import torch
from torch.library import opcheck

from speculators.losses import eager, resolve_loss_config
from speculators.losses import fused_linear as fl
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import (
    compute_metrics,
    compute_metrics_from_hidden,
)
from tests.unit.models.test_fused_losses import (
    DEVICE,
    requires_accelerator,
    requires_cuda,
)

pytestmark = pytest.mark.usefixtures("narrowest_tile")

NAMES = ("kl_div", "rkl", "jsd", "ce", "tv", "nla", "lk_hybrid")
EAGER = {
    "kl_div": eager.kl_div_loss,
    "rkl": eager.reverse_kl_div_loss,
    "jsd": eager.js_div_loss,
    "ce": eager.ce_loss,
    "tv": eager.tv_loss,
    "nla": eager.neg_log_acceptance_loss,
    "lk_hybrid": eager.lk_hybrid_loss,
}
ALL_TERMS = 0b11111
DEFERRED_TERMS = 0b10001  # kl_div + tv: deferred mode holds one jac per term


def _inputs(rows, hidden, vocab, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device=device) * scale

    weight = randn(vocab, hidden, scale=2.0 / hidden**0.5)
    verifier_hidden = randn(rows, hidden)
    h = verifier_hidden + randn(rows, hidden, scale=0.5)
    targets = verifier_hidden @ (weight + randn(vocab, hidden, scale=hidden**-0.5)).T
    return h.to(dtype), weight.to(dtype), targets.to(dtype)


# ---------------------------------------------------------------------------
# CPU: torch reference path
# ---------------------------------------------------------------------------


def test_values_match_eager_on_materialized_logits():
    h, w, t = _inputs(13, 6, 17, torch.float64, "cpu")
    out = fl.fused_linear_compound(h, w, t, NAMES, metric_terms=("tv",), chunk_rows=5)
    logits = h @ w.T
    for name in NAMES:
        # eager computes its softmaxes in fp32
        ref = EAGER[name](logits[None], t[None])[0].double()
        torch.testing.assert_close(out.losses[name], ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.metrics["tv"], out.losses["tv"].detach())
    assert torch.equal(out.pred_ids, logits.argmax(-1))
    assert torch.equal(out.target_ids, t.argmax(-1))


@pytest.mark.parametrize("name", [n for n in NAMES if n != "lk_hybrid"])
def test_gradcheck_fp64(name):
    h, w, t = _inputs(13, 6, 17, torch.float64, "cpu", seed=1)
    h.requires_grad_(True)

    def loss(x):
        return fl.fused_linear_compound(x, w, t, (name,), chunk_rows=5).losses[name]

    assert torch.autograd.gradcheck(loss, (h,))


def test_lk_hybrid_gradient_matches_eager():
    """lk_hybrid's blend weight is detached by design, so gradcheck does not
    apply; compare with eager's gradient instead."""
    h, w, t = _inputs(13, 6, 17, torch.float64, "cpu", seed=2)
    x = h.clone().requires_grad_(True)
    fl.fused_linear_compound(x, w, t, ("lk_hybrid",), chunk_rows=5).losses[
        "lk_hybrid"
    ].sum().backward()
    y = h.clone().requires_grad_(True)
    eager.lk_hybrid_loss((y @ w.T)[None], t[None]).sum().backward()
    assert x.grad is not None
    assert y.grad is not None
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)


def _opcheck_args(device, dtype, recompute_targets, deferred=False):
    h, w, t = _inputs(13, 8, 40, dtype, device, seed=3)
    target_weight = None
    if recompute_targets:
        # verifier hidden [N, H_t] and verifier lm_head [V, H_t]
        t = torch.randn(13, 6, device=device, dtype=dtype)
        target_weight = torch.randn(40, 6, device=device, dtype=dtype)
    _, _, stats, _ = torch.ops.speculators.fused_linear_loss_fwd(
        h, w, t, target_weight, ALL_TERMS, ALL_TERMS, 5, False, False
    )
    grad = torch.randn(5, 13, device=device, dtype=stats.dtype)
    fwd = (
        h.clone().requires_grad_(True),
        w,
        t,
        target_weight,
        ALL_TERMS,
        DEFERRED_TERMS if deferred else ALL_TERMS,
        5,
        deferred,
        False,
    )
    bwd = (grad, h, w, t, target_weight, stats, ALL_TERMS, 5, False)
    return fwd, bwd


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("recompute_targets", [False, True])
def test_opcheck_cpu(recompute_targets, deferred):
    fwd, bwd = _opcheck_args("cpu", torch.float64, recompute_targets, deferred)
    opcheck(torch.ops.speculators.fused_linear_loss_fwd.default, fwd)
    opcheck(torch.ops.speculators.fused_linear_loss_bwd.default, bwd)


def test_rejects_trainable_weight_and_misaligned_targets():
    h, w, t = _inputs(8, 4, 10, torch.float32, "cpu")
    with pytest.raises(ValueError, match="frozen"):
        fl.fused_linear_compound(h, w.requires_grad_(True), t, ("kl_div",))
    with pytest.raises(ValueError, match="do not match"):
        fl.fused_linear_compound(h, w.detach(), t[:-1], ("kl_div",))


def test_chunk_rows_for(monkeypatch):
    monkeypatch.delenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", raising=False)
    # 4 GiB / (151936 * (4 + 2) B) fits 4710 rows; capped at 2048
    assert fl.chunk_rows_for(151936, torch.bfloat16, 16384) == 2048
    assert fl.chunk_rows_for(151936, torch.bfloat16, 100) == 100
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    assert fl.chunk_rows_for(151936, torch.bfloat16, 16384) == 64


def test_fused_linear_is_dflash_only():
    with pytest.raises(ValueError, match="only supported by DFlash"):
        resolve_loss_config("kl_div", "fused_linear")
    train, val = DFlashDraftModel.get_trainer_kwargs(
        loss_fn='{"kl_div": 0.5, "tv": 0.5}', loss_implementation="fused_linear"
    )
    assert train["fused_linear"]
    assert val["fused_linear"]
    assert train["loss_config"] == resolve_loss_config('{"kl_div": 0.5, "tv": 0.5}')
    train, _ = DFlashDraftModel.get_trainer_kwargs(loss_fn="kl_div")
    assert not train["fused_linear"]


# ---------------------------------------------------------------------------
# GPU: Triton kernels
# ---------------------------------------------------------------------------

TERM_SETS = [("kl_div",), ("ce",), ("kl_div", "tv"), ("jsd", "rkl", "ce", "tv")]


def _fused():
    return pytest.importorskip("speculators.losses.fused")


def _materialized(logits, targets, names, **kwargs):
    """The fused path on materialized ``[N, V]`` logits, as ``[N]`` outputs."""
    out = _fused().fused_compound(logits[None], targets[None], names, **kwargs)
    return type(out)(
        losses={k: v[0] for k, v in out.losses.items()},
        metrics={k: v[0] for k, v in out.metrics.items()},
        pred_ids=out.pred_ids[0],
        target_ids=out.target_ids[0],
    )


def _run(fn, h, names):
    leaf = h.detach().clone().requires_grad_(True)
    out = fn(leaf)
    weights = torch.linspace(0.5, 1.5, len(names), device=h.device)
    total = sum(
        weights[i]
        * (out.losses[name] * (torch.arange(h.shape[0], device=h.device) % 3)).sum()
        for i, name in enumerate(names)
    )
    total.backward()
    assert leaf.grad is not None
    return out, leaf.grad


@requires_accelerator
@pytest.mark.parametrize("names", TERM_SETS, ids=["+".join(n) for n in TERM_SETS])
@pytest.mark.parametrize("vocab", [1000, 151936])
def test_matches_materialized_fused_fp32(names, vocab):
    h, w, t = _inputs(300, 64, vocab, torch.float32, DEVICE, seed=4)
    out, grad = _run(
        lambda x: fl.fused_linear_compound(
            x, w, t, names, metric_terms=("tv",), chunk_rows=128
        ),
        h,
        names,
    )
    ref, ref_grad = _run(
        lambda x: _materialized(x @ w.T, t, names, metric_terms=("tv",)),
        h,
        names,
    )
    for name in names:
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(
        out.metrics["tv"], ref.metrics["tv"], rtol=1e-4, atol=1e-5
    )
    assert torch.equal(out.pred_ids, ref.pred_ids)
    assert torch.equal(out.target_ids, ref.target_ids)
    atol = 1e-4 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=atol)


@requires_accelerator
@pytest.mark.parametrize("names", TERM_SETS, ids=["+".join(n) for n in TERM_SETS])
def test_bf16_at_least_as_accurate_as_materialized(names):
    """fp32 logits from the GEMM accumulator vs today's bf16 logits."""
    h, w, t = _inputs(256, 128, 8000, torch.bfloat16, DEVICE, seed=5)
    ref, ref_grad = _run(
        lambda x: _materialized(x.float() @ w.float().T, t, names),
        h.float(),
        names,
    )
    linear, linear_grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, names, chunk_rows=64), h, names
    )
    today, today_grad = _run(lambda x: _materialized(x @ w.T, t, names), h, names)

    def err(a, b):
        return float((a.float() - b).abs().max() / b.abs().max())

    for name in names:
        assert (
            err(linear.losses[name], ref.losses[name])
            <= 1.1 * err(today.losses[name], ref.losses[name]) + 1e-6
        ), name
    assert err(linear_grad, ref_grad) <= 1.1 * err(today_grad, ref_grad) + 1e-6


def _loss_head_peak(fn, h, names) -> int:
    _run(fn, h, names)  # Triton autotuning allocates benchmark buffers
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _run(fn, h, names)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


@requires_cuda
def test_never_materializes_draft_logits():
    h, w, t = _inputs(8192, 512, 32000, torch.bfloat16, "cuda", seed=6)
    names = ("kl_div",)
    nv_bytes = t.nbytes  # one [N, V] bf16 tensor
    chunk = 512
    linear = _loss_head_peak(
        lambda x: fl.fused_linear_compound(x, w, t, names, chunk_rows=chunk), h, names
    )
    materialized = _loss_head_peak(lambda x: _materialized(x @ w.T, t, names), h, names)
    assert materialized >= 2 * nv_bytes  # draft logits + grad_in
    # One fp32 logits tile + one bf16 dz tile, plus d hidden and per-row
    # outputs (~18 MiB here); no [N, V] tensor. Measured 112 MiB on MI355X.
    chunk_scratch = chunk * 32000 * (4 + 2)
    assert linear < 1.1 * chunk_scratch + 32 * 1024**2


@requires_cuda
def test_no_host_syncs():
    h, w, t = _inputs(512, 64, 4000, torch.bfloat16, "cuda", seed=7)
    x = h.clone().requires_grad_(True)
    fl.fused_linear_compound(x, w, t, ("kl_div",), chunk_rows=128)  # autotune first
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = fl.fused_linear_compound(x, w, t, ("kl_div", "tv"), chunk_rows=128)
        (out.losses["kl_div"].sum() + out.losses["tv"].sum()).backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@requires_accelerator
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("recompute_targets", [False, True])
def test_opcheck_gpu(recompute_targets, deferred):
    fwd, bwd = _opcheck_args(DEVICE, torch.float32, recompute_targets, deferred)
    opcheck(torch.ops.speculators.fused_linear_loss_fwd.default, fwd)
    opcheck(torch.ops.speculators.fused_linear_loss_bwd.default, bwd)


@requires_accelerator
def test_compiles_fullgraph_without_recompiling_on_new_row_counts():
    _, w, _ = _inputs(8, 64, 2000, torch.float32, DEVICE)

    def step(x, t):
        out = fl.fused_linear_compound(x, w, t, ("kl_div", "nla"), chunk_rows=128)
        return out.losses["kl_div"].sum() + out.losses["nla"].sum()

    compiled = torch.compile(step, fullgraph=True, dynamic=True, backend="aot_eager")
    for rows in (256, 320, 448):
        h, _, t = _inputs(rows, 64, 2000, torch.float32, DEVICE, seed=rows)
        x = h.requires_grad_(True)
        if rows == 320:
            torch._dynamo.config.error_on_recompile = True
        try:
            compiled(x, t).backward()
        finally:
            torch._dynamo.config.error_on_recompile = False
        assert x.grad is not None


@requires_accelerator
def test_dflash_metrics_from_hidden_match_materialized():
    h, w, t = _inputs(64, 32, 1000, torch.float32, DEVICE, seed=8)
    loss_mask = torch.ones(1, 64, device=DEVICE, dtype=torch.long)
    loss_mask[0, 10:14] = 0
    config = resolve_loss_config('{"kl_div": 0.5, "tv": 0.5}')
    kwargs: dict[str, Any] = {
        "block_size": 8,
        "loss_config": config,
        "per_position_loss_weight": "dpace",
    }
    loss, metrics = compute_metrics_from_hidden(
        h[None], w, t[None], loss_mask, chunk_rows=24, **kwargs
    )
    ref_loss, ref = compute_metrics((h @ w.T)[None], t[None], loss_mask, **kwargs)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-6)
    assert metrics.keys() == ref.keys()
    for key, value in ref.items():
        torch.testing.assert_close(metrics[key], value, rtol=1e-4, atol=1e-5, msg=key)


# ---------------------------------------------------------------------------
# GPU edge cases
# ---------------------------------------------------------------------------

_EDGE_NAMES = ("kl_div", "ce", "tv")


@requires_accelerator
@pytest.mark.parametrize("chunk_rows", [1, 7, 64, 100, 1000])
def test_results_do_not_depend_on_chunking(chunk_rows):
    """Every row is independent: chunk size (incl. a last partial chunk and a
    chunk larger than N) only changes the GEMM tiling."""
    h, w, t = _inputs(100, 64, 3000, torch.float32, DEVICE, seed=9)
    ref, ref_grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, _EDGE_NAMES, chunk_rows=100),
        h,
        _EDGE_NAMES,
    )
    out, grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, _EDGE_NAMES, chunk_rows=chunk_rows),
        h,
        _EDGE_NAMES,
    )
    for name in _EDGE_NAMES:
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-5, atol=1e-6
        )
    assert torch.equal(out.pred_ids, ref.pred_ids)
    assert torch.equal(out.target_ids, ref.target_ids)
    # fp32 rounding only: a 1- or 7-row GEMM accumulates in another order
    atol = 1e-5 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=atol)


@requires_accelerator
def test_rows_with_zero_upstream_gradient_get_exactly_zero():
    h, w, t = _inputs(96, 32, 2000, torch.bfloat16, DEVICE, seed=10)
    x = h.clone().requires_grad_(True)
    out = fl.fused_linear_compound(x, w, t, _EDGE_NAMES, chunk_rows=32)
    keep = torch.zeros(96, device=DEVICE)
    keep[40:50] = 1.0  # all other rows masked, including two whole chunks
    torch.stack([(out.losses[n] * keep).sum() for n in _EDGE_NAMES]).sum().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:40]) == 0
    assert torch.count_nonzero(x.grad[50:]) == 0
    assert torch.count_nonzero(x.grad[40:50]) > 0
    assert torch.isfinite(x.grad).all()


@requires_accelerator
def test_non_contiguous_inputs():
    h, w, t = _inputs(64, 32, 1500, torch.float32, DEVICE, seed=11)
    # same values, transposed storage (column-major) for hidden and targets
    h_strided = h.T.contiguous().T
    t_strided = t.T.contiguous().T
    assert not h_strided.is_contiguous()
    assert not t_strided.is_contiguous()
    ref, ref_grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, _EDGE_NAMES, chunk_rows=16),
        h,
        _EDGE_NAMES,
    )
    out, grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t_strided, _EDGE_NAMES, chunk_rows=16),
        h_strided,
        _EDGE_NAMES,
    )
    for name in _EDGE_NAMES:
        torch.testing.assert_close(out.losses[name], ref.losses[name])
    torch.testing.assert_close(grad, ref_grad)


@requires_accelerator
def test_saturated_logits_match_materialized():
    """Logits in the hundreds: point-mass softmaxes, log-space underflow."""
    h, w, t = _inputs(64, 32, 1000, torch.float32, DEVICE, seed=12)
    h = h * 60.0
    t = t * 60.0
    names = ("kl_div", "rkl", "jsd", "ce", "tv")
    out, grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, names, chunk_rows=16), h, names
    )
    ref, ref_grad = _run(lambda x: _materialized(x @ w.T, t, names), h, names)
    for name in names:
        assert torch.isfinite(out.losses[name]).all(), name
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-4, atol=1e-4
        )
    assert torch.isfinite(grad).all()
    atol = 1e-4 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=atol)


@requires_accelerator
def test_deterministic_across_runs():
    h, w, t = _inputs(300, 64, 8000, torch.bfloat16, DEVICE, seed=13)
    names = ("kl_div", "tv")
    first, first_grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, names, chunk_rows=128), h, names
    )
    second, second_grad = _run(
        lambda x: fl.fused_linear_compound(x, w, t, names, chunk_rows=128), h, names
    )
    for name in names:
        assert torch.equal(first.losses[name], second.losses[name])
    assert torch.equal(first_grad, second_grad)


@requires_cuda
@pytest.mark.parametrize(
    ("names", "grad_mode", "keeps_targets"),
    [
        (("ce",), "recompute", False),
        (("kl_div",), "recompute", True),
        (("kl_div",), "deferred", False),
        (("kl_div", "tv"), "deferred", False),
    ],
)
def test_backward_keeps_targets_only_when_it_needs_them(
    names, grad_mode, keeps_targets
):
    """Recompute backward needs the target logits for distribution terms;
    CE needs only the cached target argmax, and deferred mode needs nothing
    but jac."""
    h, w, _ = _inputs(2048, 256, 8000, torch.bfloat16, "cuda", seed=14)
    x = h.clone().requires_grad_(True)
    targets = torch.randn(2048, 8000, device="cuda", dtype=torch.bfloat16)
    out = fl.fused_linear_compound(
        x, w, targets, names, metric_terms=("tv",), grad_mode=grad_mode
    )
    loss = out.losses[names[0]].sum()
    nbytes = targets.nbytes
    del out
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    del targets
    freed = before - torch.cuda.memory_allocated()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    if keeps_targets:
        assert freed < nbytes // 2
    else:
        assert freed >= nbytes


def test_mismatched_dtypes_without_autocast_raise():
    h, w, t = _inputs(8, 4, 10, torch.float32, "cpu")
    with pytest.raises(ValueError, match="dtypes differ"):
        fl.fused_linear_compound(h, w.double(), t, ("kl_div",))


@requires_accelerator
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("logits_dtype", ["input", "float32"])
def test_autocast_uses_one_operand_dtype_in_both_directions(compiled, logits_dtype):
    """The trainer runs under autocast(bf16) with fp32 params (DDP). Forward
    and the backward recompute must see the same bf16 operands; result is
    bitwise that of passing the bf16 operands explicitly."""
    h, w, t = _inputs(200, 64, 3000, torch.float32, DEVICE, seed=15)
    t = t.bfloat16()
    names = ("kl_div", "tv")

    def under_autocast(x):
        with torch.autocast(str(DEVICE), dtype=torch.bfloat16):
            return fl.fused_linear_compound(
                x, w, t, names, chunk_rows=64, logits_dtype=logits_dtype
            )

    fn = (
        torch.compile(under_autocast, fullgraph=True, backend="aot_eager")
        if compiled
        else under_autocast
    )
    out, grad = _run(fn, h, names)
    explicit, explicit_grad = _run(
        lambda x: fl.fused_linear_compound(
            x.bfloat16(),
            w.bfloat16(),
            t,
            names,
            chunk_rows=64,
            logits_dtype=logits_dtype,
        ),
        h,
        names,
    )
    for name in names:
        assert torch.equal(out.losses[name], explicit.losses[name]), name
    assert torch.equal(grad, explicit_grad)
    assert grad.dtype == torch.float32

    if logits_dtype != "float32":
        return

    # fp32 tiles: consistent with the materialized loss on those fp32 logits
    def materialized(x):
        # bf16 products are exact in fp32: same math as the out_dtype GEMM,
        # but differentiable
        logits = x.bfloat16().float() @ w.bfloat16().float().T
        return _materialized(logits, t, names)

    ref, ref_grad = _run(materialized, h, names)
    for name in names:
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-5, atol=1e-6
        )
    # d hidden is formed in bf16 (dz and dz @ W), as on today's path; the
    # reference is fp32 throughout, so allow one bf16 ulp of the largest entry
    atol = 4e-3 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-2, atol=atol)
