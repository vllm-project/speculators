"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).

Two acceptance lengths are reported, both counting the verifier's bonus token:
``accept_len`` from that analytical rate (acceptance under rejection sampling)
and ``eal`` from greedy argmax matches (acceptance at temperature 0, and the
value comparable across the DFlash family).
"""

from collections.abc import Callable
from functools import partial

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
)
from speculators.models.dflash.utils import block_positions_and_loss_decay
from speculators.models.metrics import (
    compute_analytical_block_acceptance,
    compute_greedy_block_metrics,
)

__all__ = [
    "compute_metrics",
]

_EPS = 1e-8


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    pos_idx, decay_fn = block_positions_and_loss_decay(
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )

    # With an all-fused config the TV behind accept_rate and both argmaxes
    # come from the loss's own kernel pass (TV as a metric-only term when the
    # loss does not use it); otherwise they are computed separately below.
    out = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        metric_terms=("tv",),
        with_ids=True,
    )
    loss, term_losses = out.loss, out.term_losses

    # Analytical per-position acceptance rate = distributional overlap
    # = 1 - TV; the fused kernel avoids the two full-vocab fp32 softmaxes.
    accept_rate = 1.0 - out.metric_terms["tv"]
    metrics, accept_prefix, draft_mask = compute_analytical_block_acceptance(
        accept_rate, loss_mask, block_size, sample_from_anchor
    )
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        # D-PACE expects token CE, so keep the confidence BCE on fixed decay.
        confidence_decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, confidence_decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum()
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            start_pos = 0 if sample_from_anchor else 1
            conf_prefix = (
                conf_prob.reshape(-1, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum()

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    metrics.update(
        compute_greedy_block_metrics(
            out.pred_ids == out.target_ids, loss_mask, block_size, sample_from_anchor
        )
    )
    return loss, metrics
