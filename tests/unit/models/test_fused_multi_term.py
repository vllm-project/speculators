"""Multi-term fused losses: several terms from one kernel pass.

``fused_terms`` computes the draft/target softmax statistics once and every
requested term in the same stream; backward writes one gradient, the per-row
``grad_output``-weighted sum of each term's closed form. These tests check it
against the single-term kernels (fp32, tight) and the eager references (bf16),
and that ``compound_loss`` routes an all-fused config through one call with
the same result as the per-term path, D-PACE weighting included.
"""

from functools import partial

import pytest
import torch

from speculators.losses import compound_loss, eager, loss_function, resolve_loss_config
from speculators.losses.utils import dpace_loss_decay
from tests.unit.models.test_fused_losses import (
    CE_BF16_GRAD_TOL,
    CE_BF16_LOSS_TOL,
    DEVICE,
    GRAD_TOL,
    LOSS_TOL,
    requires_accelerator,
    requires_cuda,
)

EAGER = {
    "kl_div": eager.kl_div_loss,
    "rkl": eager.reverse_kl_div_loss,
    "jsd": eager.js_div_loss,
    "ce": eager.ce_loss,
    "tv": eager.tv_loss,
}
SINGLE = {
    "kl_div": "fused_kl_div_loss",
    "rkl": "fused_reverse_kl_div_loss",
    "jsd": "fused_js_div_loss",
    "ce": "fused_ce_loss",
    "tv": "fused_tv_loss",
}
SUBSETS = [
    ("kl_div", "tv"),
    ("ce", "tv"),
    ("rkl", "jsd"),
    ("jsd", "ce"),
    ("kl_div", "rkl", "jsd", "ce", "tv"),
]


def _fused():
    return pytest.importorskip("speculators.losses.fused")


def _inputs(rows: int, vocab: int, dtype: torch.dtype, seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = torch.randn(1, rows, vocab, device=DEVICE, generator=gen) * 3
    targets = torch.randn(1, rows, vocab, device=DEVICE, generator=gen) * 3
    # repeated maxima in one target row: CE must pick the first index
    targets[0, 0, 5] = targets[0, 0, 9] = targets[0, 0].max() + 1
    grad_out = torch.randn(len(EAGER), 1, rows, device=DEVICE, generator=gen)
    grad_out[:, :, ::3] = 0.0  # rows taking the backward early-out
    return logits.to(dtype), targets.to(dtype), grad_out


def _weighted_grad(fn_per_term, logits, terms, grad_out):
    """Gradient of sum_t sum_rows go_t * loss_t; ``fn_per_term`` yields losses."""
    leaf = logits.detach().clone().requires_grad_(True)
    losses = fn_per_term(leaf)
    total = sum((losses[t].float() * grad_out[i]).sum() for i, t in enumerate(terms))
    total.backward()
    assert leaf.grad is not None
    return {t: v.detach().float() for t, v in losses.items()}, leaf.grad.float()


@requires_accelerator
@pytest.mark.parametrize("terms", SUBSETS, ids=["+".join(s) for s in SUBSETS])
@pytest.mark.parametrize("vocab", [512, 151936])
def test_multi_term_matches_single_terms(terms, vocab):
    """One pass == each term's own kernel, value and summed gradient (fp32)."""
    fused = _fused()
    logits, targets, grad_out = _inputs(8, vocab, torch.float32, seed=0)

    losses, grad = _weighted_grad(
        lambda x: fused.fused_terms(x, targets, terms), logits, terms, grad_out
    )
    ref_grad = torch.zeros_like(grad)
    for i, term in enumerate(terms):
        single = getattr(fused, SINGLE[term])
        ref_losses, term_grad = _weighted_grad(
            lambda x, single=single, term=term: {term: single(x, targets)},
            logits,
            (term,),
            grad_out[i : i + 1],
        )
        # Equal up to fp32 summation order: the two launches may autotune to
        # different tiles, which reorders the blockwise reductions.
        torch.testing.assert_close(losses[term], ref_losses[term], rtol=1e-5, atol=1e-5)
        ref_grad += term_grad
    atol = 1e-5 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=atol)


@requires_accelerator
@pytest.mark.parametrize("terms", SUBSETS, ids=["+".join(s) for s in SUBSETS])
def test_multi_term_matches_eager_bf16(terms):
    """bf16, the training dtype: every term and the combined gradient vs eager."""
    fused = _fused()
    logits, targets, grad_out = _inputs(64, 512, torch.bfloat16, seed=1)
    with_ce = "ce" in terms

    losses, grad = _weighted_grad(
        lambda x: fused.fused_terms(x, targets, terms), logits, terms, grad_out
    )
    ref_losses, ref_grad = _weighted_grad(
        lambda x: {t: EAGER[t](x, targets) for t in terms}, logits, terms, grad_out
    )
    for term in terms:
        tol = CE_BF16_LOSS_TOL if term == "ce" else LOSS_TOL
        torch.testing.assert_close(
            losses[term], ref_losses[term], rtol=tol["rtol"], atol=tol["atol"]
        )
    grad_tol = CE_BF16_GRAD_TOL if with_ce else GRAD_TOL
    torch.testing.assert_close(
        grad, ref_grad, rtol=grad_tol["rtol"], atol=grad_tol["atol"]
    )


@requires_accelerator
def test_ce_ties_break_on_first_index_in_multi_term_pass():
    fused = _fused()
    logits, targets, _ = _inputs(4, 512, torch.float32, seed=2)
    ce = fused.fused_terms(logits, targets, ("ce", "tv"))["ce"]
    first = torch.argmax(targets, dim=-1)  # torch.argmax: first index on ties
    assert int(first[0, 0]) == 5
    expected = torch.logsumexp(logits, -1) - logits.gather(-1, first[..., None])[..., 0]
    torch.testing.assert_close(ce, expected, rtol=1e-5, atol=1e-5)


_COMPOUND_SPEC = '{"kl_div": 0.5, "tv": 0.3, "lk_hybrid": 0.2, "nla": 0.1}'


@requires_accelerator
@pytest.mark.parametrize("weighting", ["none", "dpace"])
def test_compound_loss_runs_one_kernel_pass(monkeypatch, weighting):
    """An all-fused config makes one kernel call and matches the per-term path."""
    fused = _fused()
    calls: list[int] = []
    apply = fused._FusedLoss.apply

    def counting_apply(*args):
        calls.append(args[2])
        return apply(*args)

    logits, targets, _ = _inputs(32, 512, torch.float32, seed=3)
    loss_mask = torch.ones(1, 32, device=DEVICE, dtype=torch.long)
    loss_mask[0, 5:9] = 0
    pos_idx = (torch.arange(32, device=DEVICE) % 8).unsqueeze(0)
    decay_fn = (
        partial(dpace_loss_decay, loss_mask=loss_mask, block_size=8, dpace_alpha=0.5)
        if weighting == "dpace"
        else None
    )
    config = resolve_loss_config(_COMPOUND_SPEC)

    leaf = logits.clone().requires_grad_(True)
    monkeypatch.setattr(fused._FusedLoss, "apply", counting_apply)
    total, term_losses = compound_loss(
        leaf, targets, loss_mask, pos_idx, config, decay_fn=decay_fn
    )
    monkeypatch.undo()
    total.backward()
    assert calls == [fused._KL.value | fused._TV.value]

    ref_leaf = logits.clone().requires_grad_(True)
    ref_total = sum(
        weight
        * loss_function(ref_leaf, targets, loss_mask, pos_idx, fn, decay_fn=decay_fn)
        for fn, weight in config.values()
    )
    ref_total.backward()  # type: ignore[union-attr]
    torch.testing.assert_close(total, ref_total, rtol=1e-5, atol=1e-6)
    assert sorted(term_losses) == [f"{n}_loss" for n in sorted(config)]
    assert leaf.grad is not None
    assert ref_leaf.grad is not None
    atol = 1e-5 * float(ref_leaf.grad.abs().max())
    torch.testing.assert_close(leaf.grad, ref_leaf.grad, rtol=1e-4, atol=atol)


@requires_accelerator
def test_compound_loss_with_an_eager_term_falls_back_per_term(monkeypatch):
    logits, targets, _ = _inputs(16, 512, torch.float32, seed=4)
    loss_mask = torch.ones(1, 16, device=DEVICE, dtype=torch.long)
    pos_idx = torch.zeros(1, 16, device=DEVICE, dtype=torch.long)
    config = {
        "kl_div": (eager.kl_div_loss, 0.5),
        "tv": (resolve_loss_config("tv")["tv"][0], 0.5),
    }
    fused = _fused()
    calls: list[int] = []
    apply = fused._FusedLoss.apply

    def counting_apply(*args):
        calls.append(args[2])
        return apply(*args)

    monkeypatch.setattr(fused._FusedLoss, "apply", counting_apply)
    total, _ = compound_loss(logits, targets, loss_mask, pos_idx, config)
    monkeypatch.undo()
    # Mixed config: the fused tv term runs alone, not via the multi-term call.
    assert calls == [fused._TV.value]
    expected = sum(
        w * loss_function(logits, targets, loss_mask, pos_idx, fn)
        for fn, w in config.values()
    )
    torch.testing.assert_close(total, expected)


@requires_cuda
def test_two_terms_allocate_one_logits_gradient():
    """kl_div + tv backward holds one grad_in, not one per term plus their sum."""
    fused = _fused()

    def backward_peak(names) -> tuple[int, int]:
        logits = torch.randn(1, 2048, 8192, device="cuda", requires_grad=True)
        targets = torch.randn_like(logits)
        out = fused.fused_compound_losses(logits, targets, names)
        loss = sum(v.sum() for v in out.values())
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base, logits.nbytes

    one, nbytes = backward_peak(("kl_div",))
    two, _ = backward_peak(("kl_div", "tv"))
    assert one >= nbytes  # the grad_in itself
    assert two - one < nbytes // 4


@requires_accelerator
def test_multi_term_compiles_fullgraph():
    fused = _fused()
    logits = torch.randn(1, 8, 512, device=DEVICE, requires_grad=True)
    targets = torch.randn(1, 8, 512, device=DEVICE)

    def loss(a, b):
        out = fused.fused_compound_losses(a, b, ("kl_div", "tv", "nla", "lk_hybrid"))
        return sum(v.sum() for v in out.values())

    torch.compile(loss, fullgraph=True)(logits, targets).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_compound_loss_with_an_empty_config_is_zero():
    logits, targets, _ = _inputs(4, 32, torch.float32, seed=5)
    loss_mask = torch.ones(1, 4, device=DEVICE, dtype=torch.long)
    pos_idx = torch.zeros(1, 4, device=DEVICE, dtype=torch.long)
    total, term_losses = compound_loss(logits, targets, loss_mask, pos_idx, {})
    assert float(total) == 0.0
    assert term_losses == {}
