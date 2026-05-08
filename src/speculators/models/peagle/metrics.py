"""Metrics and loss functions for P-EAGLE draft model."""

from typing import Any

import torch

from speculators.models.metrics import (
    compute_accuracy_multi_step,
    kl_div_loss,
    loss_function,
)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    num_depths: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute loss and accuracy metrics for P-EAGLE predictions.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len]
        anchor_pos: The starting position in the original sequence the current
            sampling chain started from. [total_sampled]
        depth: Which COD sampling round each element belongs to [total_sampled]
        seq_length: Original sequence length
        num_depths: Number of parallel depths

    Returns:
        Tuple of (loss, metrics_dict)
    """
    device = logits.device

    # TODO: batch size is always 1 for P-EAGLE; unsqueeze is only to match the
    # shared loss_function/compute_accuracy_multi_step shape contract
    orig_positions = anchor_pos + depth  # [total_sampled]
    sampled_loss_mask = loss_mask[:, orig_positions]  # [1, total_sampled]

    loss = loss_function(
        logits, targets, sampled_loss_mask, pos_idx, loss_fn=kl_div_loss
    )

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)

        # For P-EAGLE, depth serves as the position index in the speculative block.
        correct_per_pos, total_per_pos = compute_accuracy_multi_step(
            pred_ids, target_ids, sampled_loss_mask, depth.unsqueeze(0), num_depths
        )

    metrics: dict[str, Any] = {
        "loss_sum": loss.detach(),
        "loss_total": torch.tensor(1.0, device=device),
        "full_acc_sum": correct_per_pos.sum(),
        "full_acc_total": total_per_pos.sum(),
    }
    for depth in range(num_depths):
        metrics[f"position_{depth}_acc_sum"] = correct_per_pos[depth]
        metrics[f"position_{depth}_acc_total"] = total_per_pos[depth]

    return loss, metrics
