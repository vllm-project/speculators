"""Metrics and loss functions for DFlash draft model."""

from functools import partial
from typing import Any

import torch

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    domino_loss_decay,
    kl_div_loss,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    normalize_by_decay: bool = False,
    decay_mode: str = "dflash",
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_config: Mapping of ``{name: (loss_fn, weight)}``
        decay_mode: "dflash" zeros position 0; "domino" gives position 0
            weight 1.0 (for shift_label=True where position 0 is useful).

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    decay_fn = (
        partial(domino_loss_decay, gamma=gamma)
        if decay_mode == "domino"
        else partial(dflash_loss_decay, gamma=gamma)
    )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        normalize_by_decay=normalize_by_decay,
    )

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones

    acc_start = 0 if decay_mode == "domino" else 1
    metrics["full_acc_sum"] = correct_per_pos[acc_start:].sum()
    metrics["full_acc_total"] = total_per_pos[acc_start:].sum()

    for pos in range(acc_start, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
    return loss, metrics
