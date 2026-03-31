"""Metrics and loss functions for DFlash draft model."""

from typing import Any

import torch


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    block_size: int = 1,
):
    """Compute token prediction accuracy with optional block-wise breakdown.

    Compares predicted tokens (argmax of logits) against ground truth targets,
    optionally masked by loss_mask. Computes per-position
    accuracy within each block (useful for analyzing positional biases in
    block-based draft models like DFlash).

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth token IDs
            [batch, seq_len, vocab_size] or [batch, seq_len]
        loss_mask: Optional binary mask for positions to include
            [batch, seq_len]
        block_size: Size of blocks for positional accuracy breakdown (default: 1)

    Returns:
        tuple: (overall_accuracy, block_position_accuracies)
            - overall_accuracy: Float tensor, fraction correct
            - block_position_accuracies: List of per-position accuracies
              within blocks (only when block_size > 1)
    """
    target_tokens = targets
    predicted_tokens = torch.argmax(logits, dim=-1)
    correct = predicted_tokens == target_tokens

    if block_size != 1:
        accs = []
        assert loss_mask is not None  # noqa: S101
        for i in range(block_size):
            pos_cor = torch.masked_select(
                correct[:, i::block_size],
                loss_mask.to(torch.bool)[:, i::block_size],
            )
            accs.append(pos_cor.float().sum() / (pos_cor.numel() + 1e-5))

    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    correct_sum = correct.float().sum()
    full_denom = correct.numel()

    return correct_sum / (full_denom + 1e-5), accs


def loss_function(logits, target_ids, loss_mask, block_size=8, gamma=4.0):
    """Compute weighted cross-entropy loss for DFlash draft model.

    Applies exponential decay weighting based on position within blocks,
    with anchor positions (first in each block) weighted to zero.

    Args:
        logits: Model logits [B, T, V]
        target_ids: Target token IDs [B, T], -100 for ignore
        loss_mask: Binary mask [B, T]
        block_size: Size of blocks for position weighting (default: 8)
        gamma: Temperature for exponential decay (default: 4.0)

    Returns:
        Scalar loss tensor
    """
    B, T, V = logits.shape  # noqa: N806

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        target_ids.reshape(B * T),
        reduction="none",
        ignore_index=-100,
    ).view(B, T)

    idx = torch.arange(T, device=logits.device)
    k = (idx + 1) % block_size
    w = torch.exp(-((k - 1).clamp(min=0)).to(logits.dtype) / gamma)
    w = (w * (k != 0).to(logits.dtype)).view(1, T)

    m = loss_mask.to(logits.dtype).view(B, T)

    ce = ce * w * m

    denom = (m * w).sum(dim=1) + 1e-5
    return (ce.sum(dim=1) / denom).mean()


@torch.no_grad()
def compute_acceptance_rate(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor | None,
    block_size: int = 1,
):
    """Compute acceptance rate using EAGLE 3 criteria.

    EAGLE 3 acceptance formula: acceptance_prob = min(1, p / p_draft)
    where p is the target model's probability and p_draft is the draft
    model's probability.

    Args:
        draft_logits: Logits from the draft model [1, seq_len, vocab_size]
        target_logits: Logits from the target/verifier model [1, seq_len, vocab_size]
        loss_mask: Mask indicating which positions to include [1, seq_len]
        block_size: Size of each block for position-wise calculation

    Returns:
        If block_size == 1: overall acceptance rate
        Otherwise: (overall acceptance rate, list of per-position acceptance rates)
    """
    draft_probs = torch.nn.functional.softmax(draft_logits, dim=-1)
    target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

    draft_tokens = torch.argmax(draft_logits, dim=-1)

    draft_token_probs_from_draft = torch.gather(
        draft_probs, dim=-1, index=draft_tokens.unsqueeze(-1)
    ).squeeze(-1)

    draft_token_probs_from_target = torch.gather(
        target_probs, dim=-1, index=draft_tokens.unsqueeze(-1)
    ).squeeze(-1)

    acceptance_prob = torch.clamp(
        draft_token_probs_from_target / (draft_token_probs_from_draft + 1e-10),
        max=1.0,
    )
    accepted = acceptance_prob

    if block_size != 1:
        acc_rates = []
        assert loss_mask is not None  # noqa: S101
        for i in range(block_size):
            pos_accepted = torch.masked_select(
                accepted[:, i::block_size],
                loss_mask.to(torch.bool)[:, i::block_size],
            )
            acc_rates.append(pos_accepted.float().mean())

    if loss_mask is not None:
        accepted = torch.masked_select(accepted, loss_mask.to(torch.bool))

    overall_acceptance = accepted.float().mean()

    if block_size == 1:
        return overall_acceptance
    else:
        return overall_acceptance, acc_rates


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    block_size: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [B, T, V]
        targets: Target token IDs [B, T]
        loss_mask: Binary mask [B, T]
        block_size: Block size for per-position metrics

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
    """
    s_loss = loss_function(logits, targets, loss_mask)

    s_full_acc, per_position_acc = compute_accuracy(
        logits, targets, loss_mask, block_size
    )

    s_metrics: dict[str, Any] = {}
    s_metrics["loss"] = s_loss.detach().clone()
    s_metrics["full_acc"] = s_full_acc

    for pos in range(len(per_position_acc)):
        s_metrics[f"position {pos} acc"] = per_position_acc[pos]
    return s_loss, s_metrics
