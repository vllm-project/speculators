"""Metrics and loss functions for DFlash draft model."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from speculators.losses import (
    CompoundLossOutput,
    LossConfig,
    compound_loss,
    compound_loss_from_hidden,
    kl_div_loss,
)
from speculators.models.dflash.utils import block_positions_and_loss_decay
from speculators.models.metrics import compute_base_metrics

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
        partial(compound_loss, logits, targets, with_ids=True),
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
    chunk_rows: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict]:
    """``compute_metrics`` without materializing the draft logits.

    The loss, both argmaxes and the metrics come from the final hidden states
    and the frozen ``lm_head`` weight in row chunks
    (``loss_implementation="fused_linear"``); ``loss_config`` must be
    resolved with the ``"fused"`` implementation. The remaining keyword
    arguments are those of ``compute_metrics``.
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
        **kwargs,
    )


def _compute_metrics(
    loss_fn: Callable[..., CompoundLossOutput],
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Shared body: ``loss_fn(loss_mask, pos_idx, loss_config=..., decay_fn=...)``
    returns the loss and both argmaxes; the rest is per-position accuracy."""
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
    out = loss_fn(loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn)
    metrics = compute_base_metrics(
        out.loss,
        out.term_losses,
        out.pred_ids == out.target_ids,
        loss_mask,
        block_size,
        sample_from_anchor,
    )
    return out.loss, metrics
