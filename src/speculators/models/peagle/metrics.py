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
    loss_mask: torch.Tensor | None,
    sample_indices: list[torch.Tensor],
    all_indices: torch.Tensor,
    seq_length: int,
    num_depths: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute loss and accuracy metrics for P-EAGLE predictions.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len] (original sequence length)
        sample_indices: Per-depth COD sampling indices
        all_indices: Flattened COD sample indices
        seq_length: Original sequence length
        num_depths: Number of parallel depths (config max, not actual)

    Returns:
        Tuple of (loss, draft_tokens, metrics_dict)
    """
    device = logits.device
    total_sampled = logits.shape[1]

    pos_idx = torch.cat(
        [
            torch.full((len(indices),), depth, device=device, dtype=torch.long)
            for depth, indices in enumerate(sample_indices)
        ]
    ).unsqueeze(0)

    if loss_mask is not None:
        orig_positions = all_indices % seq_length
        sampled_loss_mask = loss_mask[:, orig_positions]
    else:
        sampled_loss_mask = torch.ones(
            1, total_sampled, device=device, dtype=torch.bool
        )

    loss = loss_function(
        logits, targets, sampled_loss_mask, pos_idx, loss_fn=kl_div_loss
    )

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)

        correct_per_pos, total_per_pos = compute_accuracy_multi_step(
            pred_ids, target_ids, sampled_loss_mask, pos_idx, num_depths
        )

    metrics: dict[str, Any] = {
        "loss": loss.detach(),
        "full_correct": correct_per_pos.sum(),
        "full_total": total_per_pos.sum(),
    }
    for depth in range(num_depths):
        metrics[f"position {depth} correct"] = correct_per_pos[depth]
        metrics[f"position {depth} total"] = total_per_pos[depth]

    first_depth_len = len(sample_indices[0])
    draft_tokens = torch.argmax(logits[:, :first_depth_len], dim=-1)

    return loss, draft_tokens, metrics
