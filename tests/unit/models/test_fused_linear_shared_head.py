"""Shared-head delta form of the fused linear losses.

Passing ``target_weight=weight`` makes the kernels take the draft logits as
the target tile plus ``(hidden - verifier_hidden) @ W.T``.
"""

import pytest
import torch

from speculators.losses import fused_linear as fl
from tests.unit.models.test_fused_linear import _run
from tests.unit.models.test_fused_losses import DEVICE, requires_accelerator

pytestmark = [requires_accelerator, pytest.mark.usefixtures("narrowest_tile")]


def _inputs(rows, hidden, vocab, noise, dtype, seed=0):
    """One head; draft hidden states are the verifier's plus ``noise``."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    w = torch.randn(vocab, hidden, generator=gen, device=DEVICE) * 4 / hidden**0.5
    vh = torch.randn(rows, hidden, generator=gen, device=DEVICE)
    h = vh + noise * torch.randn(rows, hidden, generator=gen, device=DEVICE)
    return h.to(dtype), w.to(dtype), vh.to(dtype)


CASES = [
    (("kl_div", "ce"), (), "deferred"),
    (("jsd", "rkl", "ce", "tv"), (), "recompute"),
    # CE-only backward still rebuilds the draft logits from the targets
    (("ce",), ("kl_div",), "recompute"),
]


@pytest.mark.parametrize(
    ("names", "metrics", "grad_mode"),
    CASES,
    ids=[c[2] + ":" + "+".join(c[0]) for c in CASES],
)
def test_shared_head_matches_separate_heads(names, metrics, grad_mode):
    h, w, vh = _inputs(300, 64, 3000, 0.5, torch.float32)

    def head(target_weight):
        return lambda x: fl.fused_linear_compound(
            x,
            w,
            vh,
            names,
            metric_terms=metrics,
            chunk_rows=128,
            target_weight=target_weight,
            grad_mode=grad_mode,
        )

    out, grad = _run(head(w), h, names)
    ref, ref_grad = _run(head(w.clone()), h, names)
    for name in names:
        torch.testing.assert_close(out.losses[name], ref.losses[name])
    for name in metrics:
        torch.testing.assert_close(out.metrics[name], ref.metrics[name])
    assert torch.equal(out.target_ids, ref.target_ids)
    assert (out.pred_ids == ref.pred_ids).float().mean() > 0.99
    atol = 1e-4 * float(ref_grad.abs().max())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=atol)


@pytest.mark.parametrize("name", ["kl_div", "tv"])
def test_shared_head_cuts_bf16_gradient_error(name):
    """Near agreement (acceptance ~0.85) bf16 logit rounding dominates the
    KL / TV gradient; the delta form removes most of it."""
    h, w, vh = _inputs(64, 256, 8192, 0.1, torch.bfloat16, seed=1)
    x = h.double().requires_grad_(True)
    logp = torch.log_softmax(x @ w.double().T, -1)
    logq = torch.log_softmax(vh.double() @ w.double().T, -1)
    if name == "kl_div":
        loss = (logq.exp() * (logq - logp)).sum()
    else:
        loss = (1 - torch.minimum(logp.exp(), logq.exp()).sum(-1)).sum()
    (ref,) = torch.autograd.grad(loss, x)

    def error(target_weight):
        x = h.detach().clone().requires_grad_(True)
        out = fl.fused_linear_compound(
            x,
            w,
            vh,
            (name,),
            target_weight=target_weight,
        )
        (grad,) = torch.autograd.grad(out.losses[name].sum(), x)
        return float((grad.double() - ref).norm() / ref.norm())

    plain = error(w.clone())
    assert error(w) < plain / 4
