"""Metrics and loss functions for DFlash draft model."""

from typing import Any

import torch


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,  # shape: [1, num_anchors * block_size, vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors * block_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors * block_size]
    block_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token prediction accuracy with block-wise breakdown.

    Compares predicted tokens (argmax of logits) against ground truth targets,
    masked by loss_mask. Computes per-position accuracy within each block
    (useful for analyzing positional biases in block-based draft models
    like DFlash).

    Args:
        logits: Model predictions [1, num_anchors * block_size, vocab_size]
        targets: Ground truth token IDs [1, num_anchors * block_size]
        loss_mask: Binary mask for positions to include
            [1, num_anchors * block_size]
        block_size: Size of blocks for positional accuracy breakdown (default: 1)

    Returns:
        tuple: (overall_accuracy, per_position_accuracies)
            - overall_accuracy: Scalar float tensor, fraction correct
            - per_position_accuracies: Float tensor of shape [block_size]
              with accuracy at each position within the block
    """
    predicted_tokens = torch.argmax(
        logits, dim=-1
    )  # shape: [1, num_anchors * block_size]
    correct = predicted_tokens == targets
    correct = torch.logical_and(correct, loss_mask)

    correct = correct.reshape(1, -1, block_size)  # shape: [1, num_anchors, block_size]
    loss_mask = loss_mask.reshape(
        1, -1, block_size
    )  # shape: [1, num_anchors, block_size]

    per_block_idx_sum = correct.float().sum(dim=1)  # shape: [1, block_size]
    per_block_idx_denom = loss_mask.float().sum(dim=1)  # shape: [1, block_size]

    total_sum = per_block_idx_sum.sum()  # shape: [1]
    total_denom = per_block_idx_denom.sum()  # shape: [1]

    return total_sum / (total_denom + 1e-5), (
        per_block_idx_sum / (per_block_idx_denom + 1e-5)
    ).reshape(-1)
    # shape: [1], [block_size]


def loss_function(
    logits,  # shape: [1, num_anchors*block_size, vocab_size]
    target_ids,  # shape: [1, num_anchors*block_size]
    loss_mask,  # shape: [1, num_anchors*block_size]
    block_size=8,
    gamma=4.0,
):
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

    in_block_idx = torch.arange(T, device=logits.device) % block_size
    # in_block_idx = 0 1 2 3 0 1 2 3, block_size = 4
    w = torch.exp(-((in_block_idx - 1).clamp(min=0)).to(logits.dtype) / gamma)
    # w = e^-(0 0 1 2 0 0 1 2) / gamma
    w = (w * (in_block_idx != 0).to(logits.dtype)).view(1, T)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma

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
    TODO: Properly refactor this function to be more modular and reusable

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
        for i in range(1, block_size):
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
    loss_mask: torch.Tensor,
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
    loss = loss_function(logits, targets, loss_mask)

    full_acc, per_position_acc = compute_accuracy(
        logits, targets, loss_mask, block_size
    )

    metrics: dict[str, Any] = {}
    metrics["loss"] = loss.detach().clone()
    metrics["full_acc"] = full_acc

    for pos in range(1, len(per_position_acc)):
        metrics[f"position {pos} acc"] = per_position_acc[pos]
    return loss, metrics
