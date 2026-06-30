"""Metrics and loss functions for P-EAGLE draft model."""

from typing import Any

import torch

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    kl_div_loss,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    num_depths: int,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute loss and accuracy metrics for P-EAGLE predictions.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len]
        anchor_pos: The starting position in the original sequence the current
            sampling chain started from [total_sampled]
        depth: Which COD sampling round each element belongs to [total_sampled]
        num_depths: Number of parallel depths
        loss_config: Mapping of ``{name: (loss_fn, weight)}``.

    Returns:
        Tuple of (loss, metrics_dict)
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    device = logits.device

    # TODO: batch size is always 1 for P-EAGLE; unsqueeze is only to match the
    # shared loss_function/compute_accuracy_multi_step shape contract
    orig_positions = anchor_pos + depth  # [total_sampled]
    sampled_loss_mask = loss_mask[:, orig_positions]  # [1, total_sampled]

    loss, term_losses = compound_loss(
        logits, targets, sampled_loss_mask, depth.unsqueeze(0), loss_config=loss_config
    )

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)

        # For P-EAGLE, depth serves as the position index in the speculative block.
        correct_per_pos, total_per_pos = compute_accuracy_multi_step(
            pred_ids, target_ids, sampled_loss_mask, depth.unsqueeze(0), num_depths
        )

    ones = torch.tensor(1.0, device=device)
    metrics: dict[str, Any] = {
        "loss_sum": loss.detach(),
        "loss_total": ones,
        "full_acc_sum": correct_per_pos.sum(),
        "full_acc_total": total_per_pos.sum(),
    }
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones
    for d in range(num_depths):
        metrics[f"position_{d}_acc_sum"] = correct_per_pos[d]
        metrics[f"position_{d}_acc_total"] = total_per_pos[d]

    return loss, metrics
