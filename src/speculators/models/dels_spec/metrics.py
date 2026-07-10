"""Metrics and loss functions for the DeLS-Spec local head.

Unlike DFlash/DSpark which use distillation losses (KL/CE against verifier
logits), the DeLS-Spec local head uses standard next-token prediction loss
against ground-truth token IDs.
"""

from typing import Any

import torch

from speculators.models.metrics import (
    compute_accuracy_multi_step,
    dflash_loss_decay,
)


def ntp_loss_per_position(
    logits: torch.Tensor,  # [1, T, vocab_size]
    target_ids: torch.Tensor,  # [1, T]
) -> torch.Tensor:  # [1, T]
    """Cross-entropy against hard token-ID targets (next-token prediction)."""
    batch_size, seq_len, vocab_size = logits.shape
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)


def compute_metrics(
    logits: torch.Tensor,  # [1, num_anchors*block_size, vocab_size]
    target_ids: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int = 16,
    gamma: float = 7.0,
) -> tuple[torch.Tensor, dict]:
    """Compute NTP loss and per-position accuracy for the DeLS-Spec local head.

    Args:
        logits: Local head logits.
        target_ids: Ground-truth token IDs (from input_ids).
        loss_mask: Binary mask (position 0 of each block is zeroed).
        block_size: Draft block size.
        gamma: Exponential decay rate for loss weighting.

    Returns:
        (loss, metrics_dict) where metrics_dict contains loss_sum, loss_total,
        full_acc, and per-position accuracy.
    """
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # [1, T]

    per_pos_loss = ntp_loss_per_position(logits, target_ids)  # [1, T]

    loss_mask_f = loss_mask.to(per_pos_loss.dtype)
    per_pos_loss = per_pos_loss * loss_mask_f

    decay = dflash_loss_decay(pos_idx.to(per_pos_loss.dtype), gamma)
    per_pos_loss = per_pos_loss * decay

    denom = loss_mask_f.sum(dim=1).clamp_min(1e-5)
    loss = (per_pos_loss.sum(dim=1) / denom).mean()

    pred_ids = torch.argmax(logits, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids,
        target_ids.unsqueeze(0) if target_ids.ndim == 1 else target_ids,
        loss_mask,
        pos_idx,
        block_size,
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()

    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
    return loss, metrics
