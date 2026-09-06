"""Metrics and loss functions for P-EAGLE draft model."""

from typing import Any

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    kl_div_loss,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute loss and component metrics for P-EAGLE predictions.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len]
        anchor_pos: The starting position in the original sequence the current
            sampling chain started from [total_sampled]
        depth: Which COD sampling round each element belongs to [total_sampled]
        loss_config: Mapping of ``{name: (loss_fn, weight)}``.

    Returns:
        Tuple of (loss, metrics_dict)
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    device = logits.device

    # TODO: batch size is always 1 for P-EAGLE; unsqueeze is only to match the
    # shared loss function shape contract
    orig_positions = anchor_pos + depth  # [total_sampled]
    sampled_loss_mask = loss_mask[:, orig_positions]  # [1, total_sampled]

    loss, term_losses = compound_loss(
        logits, targets, sampled_loss_mask, depth.unsqueeze(0), loss_config=loss_config
    )

    ones = torch.tensor(1.0, device=device)
    metrics: dict[str, Any] = {
        "loss_sum": loss.detach(),
        "loss_total": ones,
    }
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()
    return loss, metrics
