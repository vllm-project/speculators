"""Metrics and loss functions for JetSpec draft model."""

from functools import partial
from typing import Any

import torch

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    kl_div_loss,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def _jetspec_loss_decay(pos_idx: torch.Tensor, gamma: float):
    """Position decay for JetSpec (all positions contribute, including pos 0)."""
    return torch.exp(-pos_idx / gamma)


def compute_metrics(
    logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for JetSpec draft predictions."""
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=partial(_jetspec_loss_decay, gamma=gamma),
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
    metrics["full_acc_sum"] = correct_per_pos.sum()
    metrics["full_acc_total"] = total_per_pos.sum()

    for pos in range(block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
    return loss, metrics
