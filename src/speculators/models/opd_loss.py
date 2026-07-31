"""Acceptance-aware loss for Draft-OPD on-policy distillation.

Implements the Draft-OPD loss from arXiv:2605.29343: forward KL on accepted
positions, reverse KL on rejected positions with exponential position decay.
"""

from typing import Any

import torch

from speculators.models.metrics import (
    compute_accuracy_multi_step,
    kl_div_loss,
    reverse_kl_div_loss,
)

_EPS = 1e-5


@torch.no_grad()
def build_accept_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool = False,
) -> torch.Tensor:
    """Reconstruct per-position accept/reject mask via greedy comparison.

    Simulates speculative verification within each block: position k is
    accepted only if all positions before it in the block also matched
    (sequential acceptance via cumprod).

    Args:
        logits: Draft model logits [1, T, V].
        targets: Target model logits [1, T, V].
        block_size: Number of positions per anchor block.
        sample_from_anchor: If False (DFlash), position 0 is the anchor input
            and is forced to "match" so cumprod propagates correctly.

    Returns:
        Binary accept mask [1, T] (float). 1 = accepted, 0 = rejected.
    """
    draft_preds = logits.argmax(dim=-1)
    target_preds = targets.argmax(dim=-1)
    match = (draft_preds == target_preds).view(-1, block_size)

    if not sample_from_anchor:
        match[:, 0] = True

    accept_mask = match.cumprod(dim=-1).view(1, -1).float()
    return accept_mask


def opd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    accept_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    gamma: float = 0.8,
    lambda_acc: float = 1.0,
    lambda_rej: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the Draft-OPD acceptance-aware loss.

    Args:
        logits: Draft model logits [1, T, V].
        targets: Target model logits [1, T, V].
        accept_mask: Binary mask [1, T], 1 = accepted.
        loss_mask: Binary mask [1, T], 1 = valid position.
        block_size: Block size for position decay indexing.
        gamma: Exponential decay base for rejected positions.
        lambda_acc: Weight for accepted loss term.
        lambda_rej: Weight for rejected loss term.

    Returns:
        (loss, metrics) where loss is scalar and metrics uses _sum/_total
        convention for distributed reduction.
    """
    seq_len = logits.shape[1]
    device = logits.device

    accepted = accept_mask * loss_mask
    rejected = (1.0 - accept_mask) * loss_mask

    fwd_kl = kl_div_loss(logits, targets)
    rev_kl = reverse_kl_div_loss(logits, targets)

    n_acc = accepted.sum(dim=1).clamp(min=_EPS)
    l_acc = (fwd_kl * accepted).sum(dim=1) / n_acc

    pos_in_block = torch.arange(seq_len, device=device) % block_size
    decay = gamma ** pos_in_block.float()
    weighted_rej = rev_kl * rejected * decay
    z_rej = (rejected * decay).sum(dim=1).clamp(min=_EPS)
    l_rej = weighted_rej.sum(dim=1) / z_rej

    loss = (lambda_acc * l_acc + lambda_rej * l_rej) / (lambda_acc + lambda_rej)
    loss = loss.mean()

    ones = torch.tensor(1.0, device=device)
    metrics: dict[str, Any] = {
        "loss_sum": loss.detach().clone(),
        "loss_total": ones,
        "acc_loss_sum": l_acc.mean().detach(),
        "acc_loss_total": ones.clone(),
        "rej_loss_sum": l_rej.mean().detach(),
        "rej_loss_total": ones.clone(),
        "n_accepted_sum": accepted.sum().detach(),
        "n_accepted_total": ones.clone(),
        "n_rejected_sum": rejected.sum().detach(),
        "n_rejected_total": ones.clone(),
    }
    return loss, metrics


def compute_opd_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    accept_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    gamma: float = 0.8,
    lambda_acc: float = 1.0,
    lambda_rej: float = 1.0,
    sample_from_anchor: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute OPD loss and accuracy metrics.

    Wraps ``opd_loss`` and adds per-position accuracy and EAL, mirroring
    ``dflash.metrics.compute_metrics``.
    """
    loss, metrics = opd_loss(
        logits, targets, accept_mask, loss_mask,
        block_size, gamma, lambda_acc, lambda_rej,
    )

    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    start_pos = 0 if sample_from_anchor else 1
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()

    eal = torch.zeros((), device=logits.device)
    cum = torch.ones((), device=logits.device)
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
        acc = correct_per_pos[pos] / total_per_pos[pos].clamp(min=1.0)
        cum = cum * acc
        eal = eal + cum
    metrics["eal_sum"] = eal
    metrics["eal_total"] = ones.clone()

    return loss, metrics
