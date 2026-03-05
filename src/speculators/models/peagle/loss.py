"""
Loss functions for P-EAGLE training.

This module provides loss computation utilities for P-EAGLE, including
cross-entropy loss (default for P-EAGLE) and per-depth loss aggregation
for parallel multi-token prediction training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute cross-entropy loss for P-EAGLE training.

    Unlike EAGLE-3's KL divergence loss which operates on probability
    distributions, P-EAGLE uses standard cross-entropy loss with target
    token IDs.

    Args:
        logits: Model output logits of shape [batch, seq_len, vocab_size].
        target_ids: Target token IDs of shape [batch, seq_len].
        loss_mask: Optional boolean mask of shape [batch, seq_len] indicating
            which positions to include in loss computation.

    Returns:
        Scalar loss value averaged over valid positions.
    """
    # Reshape for cross_entropy: [batch * seq_len, vocab_size] and [batch * seq_len]
    batch, seq_len, vocab_size = logits.shape

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)

    # Compute per-token cross-entropy loss
    per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    # shape: [batch * seq_len]

    per_token_loss = per_token_loss.reshape(batch, seq_len)

    if loss_mask is not None:
        # Apply mask and average over valid positions
        per_token_loss = per_token_loss * loss_mask.float()
        denominator = loss_mask.float().sum() + 1e-5
        return per_token_loss.sum() / denominator
    else:
        return per_token_loss.mean()


def kl_div_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute KL divergence loss (same as EAGLE-3).

    This is kept for compatibility with EAGLE-3's loss computation pattern.
    P-EAGLE defaults to cross_entropy_loss but can optionally use KL divergence.

    Args:
        logits: Model output logits of shape [batch, seq_len, vocab_size].
        targets: Target logits/probabilities of shape [batch, seq_len, vocab_size].
        loss_mask: Optional boolean mask of shape [batch, seq_len].

    Returns:
        Scalar loss value averaged over valid positions.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_probs = F.softmax(targets, dim=-1)
    elementwise_loss = F.kl_div(
        log_probs, target_probs, reduction="none", log_target=False
    )

    if loss_mask is not None:
        elementwise_loss = elementwise_loss * loss_mask.unsqueeze(-1)
        denominator = loss_mask.sum(dim=1) + 1e-5
    else:
        denominator = logits.shape[1]

    batch_loss = torch.sum(elementwise_loss, dim=(1, 2)) / denominator
    return batch_loss.mean()


def per_depth_loss(
    all_logits: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    all_masks: list[torch.Tensor | None],
    loss_fn: str = "cross_entropy",
    depth_loss_weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute aggregate loss across parallel prediction depths.

    Computes loss for each depth independently and returns the weighted sum
    along with per-depth metrics.

    Args:
        all_logits: List of K tensors, one per depth. Shape varies by depth
            due to COD sampling: [batch, n_positions_k, vocab_size].
        all_targets: List of K target tensors matching all_logits shapes.
            For cross_entropy: [batch, n_positions_k] (token IDs).
            For kl_div: [batch, n_positions_k, vocab_size] (distributions).
        all_masks: List of K optional mask tensors, [batch, n_positions_k].
        loss_fn: Loss function type, one of 'cross_entropy' or 'kl_div'.
        depth_loss_weights: Optional per-depth loss weights. If None,
            all depths are weighted equally.

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains
        per-depth loss values keyed as 'loss_depth_{k}'.

    Raises:
        ValueError: If loss_fn is not a recognized type.
    """
    if loss_fn == "cross_entropy":
        _loss_fn = cross_entropy_loss
    elif loss_fn == "kl_div":
        _loss_fn = kl_div_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}. Use 'cross_entropy' or 'kl_div'.")

    num_depths = len(all_logits)
    if depth_loss_weights is None:
        depth_loss_weights = [1.0] * num_depths

    if len(depth_loss_weights) != num_depths:
        raise ValueError(
            f"depth_loss_weights length ({len(depth_loss_weights)}) "
            f"must match num_depths ({num_depths})"
        )

    device = all_logits[0].device
    total_loss = torch.tensor(0.0, device=device)
    metrics: dict[str, torch.Tensor] = {}

    for k in range(num_depths):
        if all_logits[k].numel() == 0:
            metrics[f"loss_depth_{k}"] = torch.tensor(0.0, device=device)
            continue

        depth_loss = _loss_fn(all_logits[k], all_targets[k], all_masks[k])
        weighted_loss = depth_loss_weights[k] * depth_loss
        total_loss = total_loss + weighted_loss
        metrics[f"loss_depth_{k}"] = depth_loss.detach().clone()

    metrics["loss"] = total_loss.detach().clone()
    return total_loss, metrics


@torch.no_grad()
def per_depth_accuracy(
    all_logits: list[torch.Tensor],
    all_target_ids: list[torch.Tensor],
    all_masks: list[torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    """Compute top-1 accuracy for each prediction depth.

    Args:
        all_logits: List of K logit tensors per depth.
        all_target_ids: List of K target token ID tensors per depth.
        all_masks: List of K optional mask tensors per depth.

    Returns:
        Dictionary with per-depth accuracy keyed as 'acc_depth_{k}'.
    """
    metrics: dict[str, torch.Tensor] = {}
    device = all_logits[0].device

    for k in range(len(all_logits)):
        if all_logits[k].numel() == 0:
            metrics[f"acc_depth_{k}"] = torch.tensor(0.0, device=device)
            continue

        predicted = torch.argmax(all_logits[k], dim=-1)
        correct = predicted == all_target_ids[k]

        if all_masks[k] is not None:
            correct = correct * all_masks[k].bool()
            denom = all_masks[k].float().sum() + 1e-5
        else:
            denom = torch.tensor(correct.numel(), dtype=torch.float, device=device)

        metrics[f"acc_depth_{k}"] = correct.float().sum() / denom

    return metrics
