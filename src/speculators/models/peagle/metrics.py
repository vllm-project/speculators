"""Metrics and loss functions for P-EAGLE draft model."""

from typing import Any

import torch


LOSS_EPSILON = 1e-5


def loss_function(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sample_indices: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KL divergence loss across all COD-sampled depths.

    Loss is normalized by total token count across all depths, so deeper
    depths with fewer sampled tokens naturally contribute less gradient.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len] (original sequence length)
        sample_indices: Per-depth COD sampling indices

    Returns:
        Tuple of (prediction_loss, total_correct, total_tokens)
    """
    with torch.no_grad():
        target_probs = torch.nn.functional.softmax(targets, dim=-1)

    logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    per_token_pred_loss = (
        torch.nn.functional.kl_div(
            logits_log_softmax.reshape(-1, logits.shape[-1]),
            target_probs.reshape(-1, target_probs.shape[-1]),
            reduction="none",
            log_target=False,
        )
        .sum(dim=-1)
        .reshape(logits.shape[0], logits.shape[1])
    )

    with torch.no_grad():
        pred_tokens = torch.argmax(logits, dim=-1)
        target_tokens = torch.argmax(target_probs, dim=-1)
        correct = (pred_tokens == target_tokens).float()

        if loss_mask is not None:
            while loss_mask.ndim > 2:
                loss_mask = loss_mask.squeeze(1)

    total_masked_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0
    start_idx = 0

    for _depth, indices in enumerate(sample_indices):
        num_samples = len(indices)
        end_idx = start_idx + num_samples

        depth_pred_loss = per_token_pred_loss[:, start_idx:end_idx]

        if loss_mask is not None:
            depth_loss_mask = loss_mask[:, indices]
            depth_pred_loss = depth_pred_loss * depth_loss_mask

            depth_correct = correct[:, start_idx:end_idx]
            total_correct += (depth_correct * depth_loss_mask).sum()
            total_tokens += depth_loss_mask.sum()
        else:
            total_tokens += num_samples

        total_masked_loss += depth_pred_loss.sum()

        start_idx = end_idx

    prediction_loss = total_masked_loss / (total_tokens + LOSS_EPSILON)

    return prediction_loss, total_correct, total_tokens


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    total_correct: torch.Tensor | float,
    total_tokens: torch.Tensor | float,
    all_indices: torch.Tensor,
    seq_length: int,
    para_depth: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute overall and per-depth accuracy.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len]
        total_correct: Pre-computed total correct count from loss_function
        total_tokens: Pre-computed total token count from loss_function
        all_indices: Flattened COD sample indices
        seq_length: Original sequence length
        para_depth: Number of parallel depths

    Returns:
        Tuple of (accuracy, per_position_accuracy_dict)
    """
    target_probs = torch.nn.functional.softmax(targets, dim=-1)
    pred_tokens = torch.argmax(logits, dim=-1)
    target_tokens = torch.argmax(target_probs, dim=-1)
    correct = (pred_tokens == target_tokens).float()

    if loss_mask is not None:
        accuracy = total_correct / (total_tokens + LOSS_EPSILON)
    else:
        num_tokens = correct.numel()
        accuracy = correct.sum() / (num_tokens + LOSS_EPSILON)

    per_position_accuracy = {}
    depths = all_indices // seq_length
    pred_tokens_flat = pred_tokens.squeeze(0)
    target_tokens_flat = target_tokens.squeeze(0)
    device = logits.device

    for depth in range(min(para_depth, 10)):
        depth_mask = depths == depth
        has_depth = depth_mask.sum() > 0
        if has_depth:
            depth_correct = (
                (pred_tokens_flat == target_tokens_flat).float()
                * depth_mask.float()
            ).sum()
            depth_total = depth_mask.sum().float()
            depth_accuracy = depth_correct / (depth_total + LOSS_EPSILON)
        else:
            depth_accuracy = torch.tensor(0.0, device=device)
        per_position_accuracy[f"position {depth} acc"] = depth_accuracy.detach()
        per_position_accuracy[f"position {depth} count"] = torch.tensor(
            1.0 if has_depth else 0.0, device=device
        )

    return accuracy, per_position_accuracy


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sample_indices: list[torch.Tensor],
    all_indices: torch.Tensor,
    seq_length: int,
    para_depth: int,
    prediction_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute loss and accuracy metrics for P-EAGLE predictions.

    Args:
        logits: Draft model logits [B, total_sampled, vocab_size]
        targets: Verifier logits [B, total_sampled, vocab_size]
        loss_mask: Binary mask [B, seq_len]
        sample_indices: Per-depth COD sampling indices
        all_indices: Flattened COD sample indices
        seq_length: Original sequence length
        para_depth: Number of parallel depths
        prediction_loss_weight: Weight for prediction loss

    Returns:
        Tuple of (loss, draft_tokens, metrics_dict)
    """
    prediction_loss, total_correct, total_tokens = loss_function(
        logits, targets, loss_mask, sample_indices
    )

    loss = prediction_loss_weight * prediction_loss

    with torch.no_grad():
        accuracy, per_position_accuracy = compute_accuracy(
            logits, targets, loss_mask,
            total_correct, total_tokens,
            all_indices, seq_length, para_depth,
        )

    metrics = {
        "loss": loss.detach(),
        "full_acc": accuracy.detach(),
        **per_position_accuracy,
    }

    if sample_indices is not None:
        first_depth_len = len(sample_indices[0])
        draft_tokens = torch.argmax(
            logits[:, :first_depth_len], dim=-1
        )
    else:
        draft_tokens = torch.argmax(logits[:, :seq_length], dim=-1)

    return loss, draft_tokens, metrics
