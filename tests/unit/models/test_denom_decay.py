"""Tests for the decayed-denominator loss normalization."""

from functools import partial

import torch

from speculators.losses import exp_loss_decay, loss_function
from speculators.losses.eager import kl_div_loss


def _batch():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 16)
    targets = torch.randn(2, 8, 16)
    loss_mask = torch.ones(2, 8)
    pos_idx = torch.arange(8).expand(2, 8).contiguous()
    return logits, targets, loss_mask, pos_idx


def test_denom_decay_is_a_true_weighted_mean():
    """With a constant decay the weighted mean must equal the undecayed mean.

    A decay that scales the numerator but not the denominator shrinks the
    reported loss by the decay factor; dividing by the decayed weight sum
    cancels it exactly, which is the property that makes the two normalizations
    comparable across gamma.
    """
    logits, targets, loss_mask, pos_idx = _batch()
    flat = torch.zeros_like(pos_idx)

    plain = loss_function(logits, targets, loss_mask, flat, loss_fn=kl_div_loss)
    weighted = loss_function(
        logits,
        targets,
        loss_mask,
        flat,
        loss_fn=kl_div_loss,
        decay_fn=partial(exp_loss_decay, gamma=0.5),
        denom_decay=True,
    )

    torch.testing.assert_close(plain, weighted)


def test_denom_decay_off_reproduces_the_stock_normalization():
    logits, targets, loss_mask, pos_idx = _batch()
    decay = partial(exp_loss_decay, gamma=0.5)

    stock = loss_function(
        logits, targets, loss_mask, pos_idx, loss_fn=kl_div_loss, decay_fn=decay
    )
    explicit_off = loss_function(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_fn=kl_div_loss,
        decay_fn=decay,
        denom_decay=False,
    )

    torch.testing.assert_close(stock, explicit_off)


def test_denom_decay_without_a_decay_fn_is_a_no_op():
    logits, targets, loss_mask, pos_idx = _batch()

    off = loss_function(logits, targets, loss_mask, pos_idx, loss_fn=kl_div_loss)
    on = loss_function(
        logits, targets, loss_mask, pos_idx, loss_fn=kl_div_loss, denom_decay=True
    )

    torch.testing.assert_close(off, on)
