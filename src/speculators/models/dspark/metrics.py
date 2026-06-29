"""Metrics and loss functions for DSpark draft model.

DSpark uses a composite loss:
    Loss = ce_alpha * CE + l1_alpha * L1_dist + confidence_alpha * BCE

Metrics use the _sum/_total naming convention for distributed reduction.
"""

from typing import Any

import torch
import torch.nn.functional as F


def compute_dspark_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eval_mask: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
    confidence_pred: torch.Tensor | None = None,
    aligned_target_logits: torch.Tensor | None = None,
    ce_loss_alpha: float = 1.0,
    l1_loss_alpha: float = 0.0,
    confidence_head_alpha: float = 0.0,
    loss_decay_gamma: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DSpark composite loss and metrics.

    Args:
        logits: [B, num_blocks, block_size, V] draft logits.
        targets: [B, num_blocks, block_size] target token IDs.
        eval_mask: [B, num_blocks, block_size] supervised positions mask.
        block_keep_mask: [B, num_blocks] valid block mask.
        block_size: Number of draft tokens per anchor.
        confidence_pred: [B, num_blocks, block_size] confidence logits (optional).
        aligned_target_logits: [B, num_blocks, block_size, V] target logits (optional).
        ce_loss_alpha: Weight for CE loss.
        l1_loss_alpha: Weight for L1 distribution matching loss.
        confidence_head_alpha: Weight for confidence calibration loss.
        loss_decay_gamma: Decay rate for position-wise loss weighting.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    device = logits.device
    _, _, _, vocab_size = logits.shape

    # Build loss weight mask with optional decay
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=device).view(1, 1, -1)
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights

    # CE Loss
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    flat_weights = loss_weight_mask.reshape(-1)
    loss_per_token = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
    ce_loss_num = (loss_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum()

    # L1 Distribution Matching Loss
    l1_loss_num = torch.tensor(0.0, device=device)
    l1_loss_den = torch.tensor(0.0, device=device)
    accept_rate_3d = None
    if aligned_target_logits is not None:
        draft_probs = torch.softmax(logits.float(), dim=-1)
        target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
        l1_dist_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
        l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()
        l1_loss_den = loss_weight_mask.sum()

        # Compute acceptance rate for confidence head supervision
        accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
        accept_rate_3d = accept_rate_3d.clamp_(0.0, 1.0)

    # Confidence Calibration Loss
    confidence_loss_num = torch.tensor(0.0, device=device)
    confidence_loss_den = torch.tensor(0.0, device=device)
    if confidence_pred is not None and accept_rate_3d is not None:
        confidence_targets = accept_rate_3d.detach()
        confidence_errors = F.binary_cross_entropy_with_logits(
            confidence_pred.float(),
            confidence_targets,
            reduction="none",
        ) * loss_weight_mask
        confidence_loss_num = confidence_errors.sum()
        confidence_loss_den = loss_weight_mask.sum()

    # Composite loss
    ce_loss = ce_loss_num / (ce_loss_den + 1e-6)
    l1_loss = l1_loss_num / (l1_loss_den + 1e-6) if l1_loss_den > 0 else torch.tensor(0.0, device=device)
    confidence_loss = (
        confidence_loss_num / (confidence_loss_den + 1e-6)
        if confidence_loss_den > 0
        else torch.tensor(0.0, device=device)
    )

    total_loss = (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    )

    # Accuracy metrics
    pred_ids = torch.argmax(logits, dim=-1)
    correct = (pred_ids == targets) & eval_mask

    # Per-position accuracy
    correct_per_pos = torch.zeros(block_size, device=device, dtype=torch.float32)
    total_per_pos = torch.zeros(block_size, device=device, dtype=torch.float32)
    for pos in range(block_size):
        pos_correct = correct[:, :, pos].float().sum()
        pos_total = eval_mask[:, :, pos].float().sum()
        correct_per_pos[pos] = pos_correct
        total_per_pos[pos] = pos_total

    # Build metrics dict with _sum/_total convention
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = total_loss.detach().clone()
    metrics["loss_total"] = torch.tensor(1.0, device=device)
    metrics["ce_loss_sum"] = ce_loss_num.detach().clone()
    metrics["ce_loss_total"] = ce_loss_den.detach().clone()

    # Position 0 is the anchor — excluded from accuracy
    if block_size > 1:
        metrics["full_acc_sum"] = correct_per_pos[1:].sum()
        metrics["full_acc_total"] = total_per_pos[1:].sum()
        for pos in range(1, block_size):
            metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
            metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    if l1_loss_den > 0:
        metrics["l1_loss_sum"] = l1_loss_num.detach().clone()
        metrics["l1_loss_total"] = l1_loss_den.detach().clone()

    if confidence_loss_den > 0:
        metrics["confidence_loss_sum"] = confidence_loss_num.detach().clone()
        metrics["confidence_loss_total"] = confidence_loss_den.detach().clone()

    return total_loss, metrics


__all__ = [
    "compute_dspark_metrics",
]
