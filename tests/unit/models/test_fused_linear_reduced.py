"""``fused_linear_reduced``'s in-op reduction against autograd through
``fused.compose_loss`` and the ``_weighted_sum`` reduction, for every named
loss: its hand-written term derivatives must follow ``compose_loss``. CPU
only (no kernels)."""

import pytest
import torch

from speculators.losses import fused
from speculators.losses.fused_linear_reduced import _chunk_reduction
from speculators.losses.utils import dpace_loss_decay

ROWS, BLOCK = 8, 4


@pytest.mark.parametrize("alpha", [None, 0.5], ids=["fixed-decay", "dpace"])
@pytest.mark.parametrize("name", sorted(fused.LOSS_TERMS))
def test_reduction_matches_autograd(name, alpha):
    gen = torch.Generator().manual_seed(0)
    losses = torch.rand(fused.N_TERMS, ROWS, generator=gen, dtype=torch.float64)
    mask = torch.tensor([0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.25, 1.0], dtype=torch.float64)
    decay = torch.rand(ROWS, generator=gen, dtype=torch.float64)
    denom = mask.sum() + 1e-5
    names, weights = [name, "ce"], [0.7, 0.3]

    total, mult = _chunk_reduction(
        losses, names, weights, mask, decay, denom, alpha, BLOCK
    )

    terms = {
        t: losses[slot].clone().requires_grad_(True)
        for t, slot in fused.TERM_SLOTS.items()
    }
    ref = losses.new_zeros(())
    for n, w in zip(names, weights, strict=True):
        el = fused.compose_loss(n, terms) * mask
        row_decay = decay
        if alpha is not None:
            row_decay = dpace_loss_decay(
                mask[None], mask[None], BLOCK, alpha, elementwise_loss=el.detach()[None]
            )[0]
        ref = ref + w * (el * row_decay).sum() / denom
    grads = torch.autograd.grad(ref, list(terms.values()), allow_unused=True)

    torch.testing.assert_close(total, ref.detach())
    for t, grad in zip(terms, grads, strict=True):
        expected = torch.zeros_like(mask) if grad is None else grad
        torch.testing.assert_close(mult[t], expected, msg=t)
