"""Argmax and metric-only terms from the fused loss pass.

The fused forward also returns the first-index argmax of the draft and target
rows, and can compute primitive terms that are never differentiated (DSpark's
TV for ``accept_rate``). Models take both from ``compound_loss_with_outputs``
instead of separate full-vocab passes; non-fused configs fall back to
``torch.argmax`` and the caller's own term function.
"""

from functools import partial

import pytest
import torch

from speculators.losses import compound_loss_with_outputs, resolve_loss_config
from speculators.models.dflash.metrics import compute_metrics as dflash_metrics
from speculators.models.dspark.metrics import compute_metrics as dspark_metrics
from tests.unit.models.test_fused_losses import (
    DEVICE,
    requires_accelerator,
    requires_cuda,
)

pytestmark = pytest.mark.usefixtures("narrowest_tile")


def _fused():
    return pytest.importorskip("speculators.losses.fused")


def _tie_heavy(rows: int, vocab: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Small-integer logits (many repeated maxima) plus targeted tie layouts."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randint(-3, 4, (1, rows, vocab), device=DEVICE, generator=gen).float()
    x[0, 0] = 1.0  # all equal: index 0
    x[0, 1, -1] = 9.0  # sole max in the (non-power-of-2) tail
    x[0, 2, vocab // 2] = x[0, 2, -1] = 9.0  # tie across blocks: earlier wins
    x[0, 3, 7] = x[0, 3, 3] = 9.0  # tie within a block: earlier wins
    return x.to(dtype)


TERM_SETS = [("kl_div",), ("ce",), ("kl_div", "tv"), ("jsd", "ce", "tv")]


@requires_accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("vocab", [512, 1000, 151936])
@pytest.mark.parametrize("names", TERM_SETS, ids=["+".join(s) for s in TERM_SETS])
def test_argmax_equals_torch_argmax(dtype, vocab, names):
    fused = _fused()
    logits = _tie_heavy(8, vocab, dtype, seed=0)
    targets = _tie_heavy(8, vocab, dtype, seed=1)
    out = fused.fused_compound(logits, targets, names)
    assert out.pred_ids.dtype == torch.int64
    assert torch.equal(out.pred_ids, torch.argmax(logits, dim=-1))
    assert torch.equal(out.target_ids, torch.argmax(targets, dim=-1))


@requires_accelerator
def test_metric_only_term_is_computed_but_not_differentiated():
    fused = _fused()
    torch.manual_seed(2)
    logits = torch.randn(1, 32, 4096, device=DEVICE) * 3
    targets = torch.randn(1, 32, 4096, device=DEVICE) * 3

    leaf = logits.clone().requires_grad_(True)
    out = fused.fused_compound(leaf, targets, ("kl_div",), metric_terms=("tv",))
    out.losses["kl_div"].sum().backward()
    assert not out.metrics["tv"].requires_grad
    torch.testing.assert_close(
        out.metrics["tv"], fused.fused_tv_loss(logits, targets), rtol=1e-5, atol=1e-6
    )

    ref_leaf = logits.clone().requires_grad_(True)
    fused.fused_kl_div_loss(ref_leaf, targets).sum().backward()
    assert leaf.grad is not None
    assert ref_leaf.grad is not None
    torch.testing.assert_close(leaf.grad, ref_leaf.grad, rtol=1e-5, atol=1e-9)


@requires_accelerator
def test_metric_term_shared_with_a_loss_term():
    fused = _fused()
    torch.manual_seed(3)
    logits = torch.randn(1, 16, 512, device=DEVICE, requires_grad=True)
    targets = torch.randn(1, 16, 512, device=DEVICE)
    out = fused.fused_compound(logits, targets, ("nla",), metric_terms=("tv",))
    nla = -torch.log((1.0 - out.metrics["tv"]).clamp_min(1e-5))
    torch.testing.assert_close(out.losses["nla"].detach(), nla)


@requires_cuda
def test_metric_tv_does_not_keep_targets_for_a_ce_loss():
    """Only differentiated terms decide what backward keeps alive."""
    fused = _fused()
    logits = torch.randn(1, 1024, 4096, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    targets = torch.randn_like(logits)
    out = fused.fused_compound(logits, targets, ("ce",), metric_terms=("tv",))
    loss = out.losses["ce"].sum()
    nbytes = targets.nbytes
    del targets, out
    held = torch.cuda.memory_allocated() - base
    loss.backward()
    assert held < nbytes // 2


def _loss_inputs(rows: int, vocab: int, device):
    gen = torch.Generator(device=device).manual_seed(4)
    logits = torch.randn(1, rows, vocab, device=device, generator=gen) * 2
    targets = logits + torch.randn(1, rows, vocab, device=device, generator=gen)
    loss_mask = torch.ones(1, rows, device=device, dtype=torch.long)
    loss_mask[0, 3] = 0
    pos_idx = (torch.arange(rows, device=device) % 8).unsqueeze(0)
    return logits, targets, loss_mask, pos_idx


def test_compound_loss_with_outputs_falls_back_for_eager():
    logits, targets, loss_mask, pos_idx = _loss_inputs(16, 64, "cpu")
    config = resolve_loss_config('{"kl_div": 0.5, "tv": 0.5}', "eager")
    out = compound_loss_with_outputs(
        logits, targets, loss_mask, pos_idx, config, metric_terms=("tv",)
    )
    assert torch.equal(out.pred_ids, torch.argmax(logits, dim=-1))
    assert torch.equal(out.target_ids, torch.argmax(targets, dim=-1))
    assert out.metric_terms == {}
    assert sorted(out.term_losses) == ["kl_div_loss", "tv_loss"]


@requires_accelerator
def test_compound_loss_with_outputs_fused():
    fused = _fused()
    logits, targets, loss_mask, pos_idx = _loss_inputs(16, 512, DEVICE)
    config = resolve_loss_config("kl_div")
    out = compound_loss_with_outputs(
        logits, targets, loss_mask, pos_idx, config, metric_terms=("tv",)
    )
    assert torch.equal(out.pred_ids, torch.argmax(logits, dim=-1))
    assert torch.equal(out.target_ids, torch.argmax(targets, dim=-1))
    torch.testing.assert_close(
        out.metric_terms["tv"],
        fused.fused_tv_loss(logits, targets),
        rtol=1e-5,
        atol=1e-6,
    )


def _raise(*_args):
    raise AssertionError("tv_loss_fn must not run when the loss pass provides TV")


@requires_accelerator
@pytest.mark.parametrize("spec", ["kl_div", '{"ce": 0.1, "tv": 0.9}'])
def test_dspark_takes_accept_rate_and_ids_from_the_loss_pass(spec):
    fused = _fused()
    logits, targets, loss_mask, _ = _loss_inputs(32, 512, DEVICE)
    call = partial(dspark_metrics, logits, targets, None, loss_mask, 8)
    _, metrics = call(resolve_loss_config(spec), tv_loss_fn=_raise)
    # eager config: separate tv_loss_fn and torch.argmax (the fallback path)
    _, ref = call(resolve_loss_config(spec, "eager"), tv_loss_fn=fused.fused_tv_loss)
    for key in ("accept_rate_sum", "accept_len_sum", "full_acc_sum", "eal_sum"):
        torch.testing.assert_close(metrics[key], ref[key], rtol=1e-5, atol=1e-5)


@requires_accelerator
def test_dflash_accuracy_from_the_loss_pass_matches_eager():
    logits, targets, loss_mask, _ = _loss_inputs(32, 512, DEVICE)
    _, metrics = dflash_metrics(
        logits,
        targets,
        loss_mask,
        block_size=8,
        loss_config=resolve_loss_config("kl_div"),
    )
    _, ref = dflash_metrics(
        logits,
        targets,
        loss_mask,
        block_size=8,
        loss_config=resolve_loss_config("kl_div", "eager"),
    )
    for key in ref:
        if "acc" in key or key.startswith("eal"):
            assert torch.equal(metrics[key], ref[key]), key


@requires_accelerator
def test_fused_compound_compiles_fullgraph():
    fused = _fused()
    logits = torch.randn(1, 8, 512, device=DEVICE, requires_grad=True)
    targets = torch.randn(1, 8, 512, device=DEVICE)

    def step(a, b):
        out = fused.fused_compound(a, b, ("kl_div",), metric_terms=("tv",))
        hits = (out.pred_ids == out.target_ids).float().sum()
        return out.losses["kl_div"].sum() + 0 * hits, out.metrics["tv"]

    loss, tv = torch.compile(step, fullgraph=True)(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(tv).all()
