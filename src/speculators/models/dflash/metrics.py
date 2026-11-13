"""Metrics and loss functions for DFlash draft model."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from speculators.losses import (
    CompoundLossOutput,
    LossConfig,
    compound_loss_from_hidden,
    compound_loss_with_outputs,
    dflash_loss_decay,
    dpace_loss_decay,
    kl_div_loss,
)
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

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
    return _compute_metrics(
        partial(compound_loss_with_outputs, logits, targets),
        loss_mask,
        block_size=block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )


def compute_metrics_from_hidden(
    hidden: torch.Tensor,  # shape: [1, num_anchors*block_size, hidden_size]
    lm_head_weight: torch.Tensor,  # shape: [draft_vocab_size, hidden_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """``compute_metrics`` without materializing the draft logits.

    The loss, both argmaxes and the metrics come from the final hidden states
    and the frozen ``lm_head`` weight in row chunks
    (``loss_implementation="fused_linear"``); ``loss_config`` must be
    resolved with the ``"fused"`` implementation.
    """
    return _compute_metrics(
        partial(
            compound_loss_from_hidden,
            hidden,
            lm_head_weight,
            targets,
            chunk_rows=chunk_rows,
        ),
        loss_mask,
        block_size=block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )


def _compute_metrics(
    loss_fn: Callable[..., CompoundLossOutput],
    loss_mask: torch.Tensor,
    block_size: int,
    gamma: float,
    loss_config: LossConfig | None,
    per_position_loss_weight: str,
    dpace_alpha: float,
    sample_from_anchor: bool,
) -> tuple[torch.Tensor, dict]:
    """Shared body: ``loss_fn(loss_mask, pos_idx, loss_config=..., decay_fn=...)``
    returns the loss and both argmaxes; the rest is per-position accuracy."""
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    seq_len = loss_mask.shape[1]
    device = loss_mask.device
    pos_idx = torch.arange(seq_len, device=device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    # The argmaxes come from the fused loss pass when the config is all fused.
    out = loss_fn(loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn)
    loss, term_losses = out.loss, out.term_losses
    pred_ids, target_ids = out.pred_ids, out.target_ids

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Start position: 0 if sample_from_anchor else 1 (skip anchor)
    start_pos = 0 if sample_from_anchor else 1
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()

    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    # Counted per block so the accepted run is formed before any averaging; the
    # sum/total pair then pools across batches and ranks like every other metric.
    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(-1, block_size)[:, start_pos:],
        loss_mask.to(torch.bool).reshape(-1, block_size)[:, start_pos:],
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total
    return loss, metrics
