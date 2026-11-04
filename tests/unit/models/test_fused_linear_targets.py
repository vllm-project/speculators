"""Fused linear losses with recomputed targets (``target_weight``).

With the verifier ``lm_head`` weight, ``targets`` is the post-norm verifier
hidden state and the target logits are recomputed per chunk, so the loss head
holds no ``[N, V]`` tensor at all. The result must equal passing the same
target logits materialized in fp32.
"""

import pytest
import torch

from speculators.losses import fused_linear as fl
from tests.unit.models.test_fused_linear import _run
from tests.unit.models.test_fused_losses import (
    DEVICE,
    requires_accelerator,
    requires_cuda,
)

NAMES = ("kl_div", "rkl", "jsd", "ce", "tv", "nla", "lk_hybrid")


def _inputs(rows, hidden, verifier_hidden, vocab, dtype, device, seed=0):
    """Draft hidden/weight plus verifier hidden/weight that give close softmaxes."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device=device) * scale

    vh = randn(rows, verifier_hidden)
    tw = randn(vocab, verifier_hidden, scale=2.0 / verifier_hidden**0.5)
    # draft side: a noisy linear image of the verifier side
    mix = randn(verifier_hidden, hidden, scale=verifier_hidden**-0.5)
    h = vh @ mix + randn(rows, hidden, scale=0.3)
    w = tw @ mix + randn(vocab, hidden, scale=0.5 / hidden**0.5)
    return h.to(dtype), w.to(dtype), vh.to(dtype), tw.to(dtype)


def _target_logits(vh, tw):
    """The target logits the recompute path forms per chunk, materialized."""
    if vh.dtype in (torch.float16, torch.bfloat16):
        return torch.mm(vh, tw.T, out_dtype=torch.float32)
    return vh @ tw.T


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def test_values_match_materialized_targets_fp64():
    h, w, vh, tw = _inputs(13, 6, 5, 17, torch.float64, "cpu")
    out = fl.fused_linear_compound(
        h, w, vh, NAMES, metric_terms=("tv",), chunk_rows=5, target_weight=tw
    )
    ref = fl.fused_linear_compound(
        h, w, _target_logits(vh, tw), NAMES, metric_terms=("tv",), chunk_rows=5
    )
    for name in NAMES:
        torch.testing.assert_close(out.losses[name], ref.losses[name])
    torch.testing.assert_close(out.metrics["tv"], ref.metrics["tv"])
    assert torch.equal(out.pred_ids, ref.pred_ids)
    assert torch.equal(out.target_ids, ref.target_ids)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "lk_hybrid"])
def test_gradcheck_fp64(name):
    h, w, vh, tw = _inputs(13, 6, 5, 17, torch.float64, "cpu", seed=1)
    h.requires_grad_(True)

    def loss(x):
        out = fl.fused_linear_compound(
            x, w, vh, (name,), chunk_rows=5, target_weight=tw
        )
        return out.losses[name]

    assert torch.autograd.gradcheck(loss, (h,))


def test_validation():
    h, w, vh, tw = _inputs(8, 4, 3, 10, torch.float32, "cpu")
    with pytest.raises(ValueError, match="must be frozen"):
        fl.fused_linear_compound(
            h, w, vh, ("kl_div",), target_weight=tw.clone().requires_grad_(True)
        )
    with pytest.raises(ValueError, match="do not match"):
        fl.fused_linear_compound(h, w, vh, ("kl_div",), target_weight=tw[:-1])
    with pytest.raises(ValueError, match="do not match"):
        fl.fused_linear_compound(h, w, vh[:, :-1], ("kl_div",), target_weight=tw)
    with pytest.raises(ValueError, match="dtypes differ"):
        fl.fused_linear_compound(h, w, vh, ("kl_div",), target_weight=tw.double())


def test_chunk_rows_count_the_target_tile(monkeypatch):
    monkeypatch.delenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", raising=False)
    # 4 GiB / (151936 * (4 + 4 + 2) B) = 2826 -> 2048 rows
    assert (
        fl.chunk_rows_for(151936, torch.bfloat16, 16384, recompute_targets=True) == 2048
    )
    assert fl.chunk_rows_for(151936, torch.bfloat16, 16384) == 4096


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

TERM_SETS = [("kl_div",), ("ce",), ("kl_div", "tv"), ("jsd", "rkl", "ce", "tv")]


@requires_accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("names", TERM_SETS, ids=["+".join(n) for n in TERM_SETS])
@pytest.mark.parametrize("vocab", [1000, 151936])
def test_matches_materialized_fp32_targets(dtype, names, vocab):
    """CE-only sets also check that backward skips the target GEMM."""
    h, w, vh, tw = _inputs(300, 64, 48, vocab, dtype, DEVICE, seed=2)
    out, grad = _run(
        lambda x: fl.fused_linear_compound(
            x, w, vh, names, metric_terms=("tv",), chunk_rows=128, target_weight=tw
        ),
        h,
        names,
    )
    logits = _target_logits(vh, tw)
    ref, ref_grad = _run(
        lambda x: fl.fused_linear_compound(
            x, w, logits, names, metric_terms=("tv",), chunk_rows=128
        ),
        h,
        names,
    )
    # same GEMM per chunk vs one full GEMM: equal up to accumulation order
    for name in names:
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-5, atol=1e-5
        )
    torch.testing.assert_close(
        out.metrics["tv"], ref.metrics["tv"], rtol=1e-5, atol=1e-5
    )
    assert torch.equal(out.pred_ids, ref.pred_ids)
    assert torch.equal(out.target_ids, ref.target_ids)
    atol = 1e-4 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=atol)


@requires_cuda
def test_loss_head_holds_no_vocab_sized_tensor():
    h, w, vh, tw = _inputs(8192, 512, 512, 32000, torch.bfloat16, "cuda", seed=3)
    names = ("kl_div", "tv")
    chunk = 512

    def head(x):
        return fl.fused_linear_compound(
            x, w, vh, names, chunk_rows=chunk, target_weight=tw
        )

    _run(head, h, names)  # Triton autotuning allocates benchmark buffers
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _run(head, h, names)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    nv_bytes = 8192 * 32000 * 2  # one [N, V] bf16 tensor
    # fp32 draft + fp32 target + bf16 dz tiles, plus per-row outputs
    chunk_scratch = chunk * 32000 * (4 + 4 + 2)
    assert peak < 1.1 * chunk_scratch + 32 * 1024**2
    assert peak < nv_bytes // 2


@requires_accelerator
def test_autocast_matches_explicit_bf16_operands():
    h, w, vh, tw = _inputs(200, 64, 48, 3000, torch.float32, DEVICE, seed=4)
    names = ("kl_div", "ce")

    def under_autocast(x):
        with torch.autocast(str(DEVICE), dtype=torch.bfloat16):
            return fl.fused_linear_compound(
                x, w, vh, names, chunk_rows=64, target_weight=tw
            )

    out, grad = _run(under_autocast, h, names)
    explicit, explicit_grad = _run(
        lambda x: fl.fused_linear_compound(
            x.bfloat16(),
            w.bfloat16(),
            vh.bfloat16(),
            names,
            chunk_rows=64,
            target_weight=tw.bfloat16(),
        ),
        h,
        names,
    )
    for name in names:
        assert torch.equal(out.losses[name], explicit.losses[name]), name
    assert torch.equal(grad, explicit_grad)


@requires_accelerator
def test_compiles_fullgraph_without_recompiling_on_new_row_counts():
    _, w, _, tw = _inputs(8, 64, 48, 2000, torch.float32, DEVICE)

    def step(x, vh):
        out = fl.fused_linear_compound(
            x, w, vh, ("kl_div", "nla"), chunk_rows=128, target_weight=tw
        )
        return out.losses["kl_div"].sum() + out.losses["nla"].sum()

    compiled = torch.compile(step, fullgraph=True, dynamic=True)
    for rows in (256, 320, 448):
        h, _, vh, _ = _inputs(rows, 64, 48, 2000, torch.float32, DEVICE, seed=rows)
        x = h.requires_grad_(True)
        if rows == 320:
            torch._dynamo.config.error_on_recompile = True
        try:
            compiled(x, vh).backward()
        finally:
            torch._dynamo.config.error_on_recompile = False
        assert x.grad is not None
