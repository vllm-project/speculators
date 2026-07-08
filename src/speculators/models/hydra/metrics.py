"""Metrics and loss functions for Hydra draft model."""

import torch

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_single_step,
    kl_div_loss,
)


def align_for_head(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    head_idx: int,
):
    """Align logits, targets, and loss_mask for a given head index.

    Head i predicts token at position t+i+1 given hidden state at position t.
    So for head i we need to shift targets by i+1 and trim logits by i from the
    end (no targets available for the last i positions).

    Args:
        logits: [1, seq_len, vocab_size]
        targets: [1, seq_len, vocab_size]
        loss_mask: [1, seq_len] or None
        head_idx: 0-based head index
    """
    shift = head_idx
    if shift > 0:
        logits = logits[:, :-shift]
        targets = targets[:, shift:]
        if loss_mask is not None:
            loss_mask = loss_mask[:, shift:]
    if loss_mask is not None:
        loss_mask = loss_mask.detach().clone()
    return logits, targets, loss_mask


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    head_idx: int,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given hydra head.

    Args:
        logits: Logits from head_idx, shape [1, seq_len, draft_vocab_size]
        targets: Verifier output distribution, shape [1, seq_len, draft_vocab_size]
        loss_mask: Mask for valid positions, shape [1, seq_len]
        head_idx: 0-based head index
        loss_config: Loss function configuration

    Returns:
        Loss value and metrics dictionary.
    """
    if loss_config is None:
        loss_config = {"kl_div": (kl_div_loss, 1.0)}

    s_logits, s_targets, s_loss_mask = align_for_head(
        logits, targets, loss_mask, head_idx
    )

    seq_len = s_logits.shape[1]
    if s_loss_mask is None:
        s_loss_mask = torch.ones(1, seq_len, device=s_logits.device, dtype=torch.bool)

    pos_idx = torch.full(
        (1, seq_len), head_idx, device=s_logits.device, dtype=torch.long
    )

    s_loss, term_losses = compound_loss(
        s_logits,
        s_targets,
        s_loss_mask,
        pos_idx,
        loss_config=loss_config,
    )

    pred_ids = torch.argmax(s_logits, dim=-1)
    target_ids = torch.argmax(s_targets, dim=-1)

    acc_mask = s_loss_mask.clone() if s_loss_mask is not None else None
    full_correct, full_total, cond_correct, cond_total = compute_accuracy_single_step(
        pred_ids,
        target_ids,
        acc_mask,
        acc_mask,
    )

    ones = torch.tensor(1.0, device=s_loss.device)
    s_metrics = {}
    s_metrics[f"loss_{head_idx}_sum"] = s_loss.detach().clone()
    s_metrics[f"loss_{head_idx}_total"] = ones
    for term_name, term_val in term_losses.items():
        s_metrics[f"{term_name}_{head_idx}_sum"] = term_val
        s_metrics[f"{term_name}_{head_idx}_total"] = ones
    s_metrics[f"full_acc_{head_idx}_sum"] = full_correct
    s_metrics[f"full_acc_{head_idx}_total"] = full_total
    s_metrics[f"cond_acc_{head_idx}_sum"] = cond_correct
    s_metrics[f"cond_acc_{head_idx}_total"] = cond_total

    return s_loss, s_metrics
