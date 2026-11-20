"""Metrics and loss functions for DFlash draft model."""

from typing import Any

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    kl_div_loss,
)
from speculators.models.dflash.utils import block_positions_and_loss_decay
from speculators.models.metrics import compute_greedy_block_metrics

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_config: Mapping of ``{name: (loss_fn, weight)}``
        per_position_loss_weight: Weighting option for per-position block-drafting loss
        dpace_alpha: Smoothing constant for D-Pace loss weighting

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
            - eal: Expected Accepted Length, the mean per-block accepted run
              plus the verifier's bonus token (headline metric)
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    pos_idx, decay_fn = block_positions_and_loss_decay(
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )

    # The argmaxes come from the fused loss pass when the config is all fused.
    out = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        with_ids=True,
    )
    loss, term_losses = out.loss, out.term_losses
    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
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
