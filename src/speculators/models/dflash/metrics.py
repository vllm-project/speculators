"""Metrics and loss functions for DFlash draft model."""

from functools import partial
from typing import Any

import torch

from speculators.models.metrics import (
    ce_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    loss_function,
)


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
    """
    seq_len = logits.shape[1]  # noqa: N806
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    loss = loss_function(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_fn=ce_loss,
        decay_fn=partial(dflash_loss_decay, gamma=gamma),
    )

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    metrics: dict[str, Any] = {}
    metrics["loss sum"] = loss.detach().clone()
    metrics["loss count"] = torch.tensor(1.0, device=logits.device)
    # Position 0 is the anchor — intentionally excluded from accuracy
    metrics["full_acc sum"] = correct_per_pos[1:].sum()
    metrics["full_acc count"] = total_per_pos[1:].sum()

    for pos in range(1, block_size):
        metrics[f"position {pos} acc sum"] = correct_per_pos[pos]
        metrics[f"position {pos} acc count"] = total_per_pos[pos]
    return loss, metrics
