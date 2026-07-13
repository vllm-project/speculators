"""Metrics and loss functions for PARD-2 draft model.

Replaces DFlash's fixed position decay with confidence-adaptive token (CAT)
weighting derived from the target model's per-token probability.
"""

from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812

from speculators.models.metrics import (
    LossConfig,
    compute_accuracy_multi_step,
    kl_div_loss,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def _compute_cat_weights(
    verifier_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute confidence-adaptive token (CAT) weights.

    For each position k within a block, CAT weight ŝ_k is the cumulative
    product of target model confidences for all preceding positions:
    ŝ_k = ∏_{j=0}^{k-1} P(y_{n+j} | ...; θ_target)

    Args:
        verifier_logits: Target model logits at aligned positions
            [1, num_anchors*block_size, vocab_size].
        target_token_ids: Ground-truth token IDs at aligned positions
            [1, num_anchors*block_size].
        block_size: Number of positions per anchor block.

    Returns:
        CAT weights with shape [1, num_anchors*block_size].
    """
    probs = F.softmax(verifier_logits.float(), dim=-1)
    token_probs = probs.gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
    # token_probs: [1, num_anchors*block_size]

    num_tokens = token_probs.shape[1]
    num_blocks = num_tokens // block_size
    block_probs = token_probs.view(1, num_blocks, block_size)

    shifted = torch.ones_like(block_probs)
    shifted[:, :, 1:] = block_probs[:, :, :-1]
    cum_probs = shifted.cumprod(dim=2)

    # Position 0 (anchor) gets weight 0 (same as DFlash)
    cum_probs[:, :, 0] = 0.0

    return cum_probs.view(1, num_tokens)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    verifier_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    ce_alpha: float = 0.1,
    kd_alpha: float = 1.0,
    kd_temperature: float = 1.0,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute PARD-2 loss with CAT weighting and CE+KD objective.

    Args:
        logits: Draft model logits [1, T, draft_vocab_size].
        targets: Target model logits [1, T, draft_vocab_size].
        loss_mask: Binary mask [1, T].
        block_size: Number of positions per anchor block.
        verifier_logits: Full target logits for CAT computation [1, T, vocab_size].
        target_token_ids: Ground-truth token IDs at aligned positions [1, T].
        ce_alpha: Weight for CE loss term.
        kd_alpha: Weight for KD loss term.
        kd_temperature: Temperature for KD softmax.
        loss_config: Base loss configuration (used for KD).

    Returns:
        Tuple of (loss, metrics_dict).
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG

    cat_weights = _compute_cat_weights(verifier_logits, target_token_ids, block_size)
    # cat_weights: [1, T], position 0 (anchor) already zeroed out
    effective_mask = loss_mask.to(logits.dtype) * cat_weights.to(logits.dtype)

    # --- KD loss: KL divergence weighted by CAT ---
    kd_loss = _kd_loss(logits, targets, effective_mask, kd_temperature)

    # --- CE loss: cross-entropy against hard targets weighted by CAT ---
    ce_loss = _ce_loss(logits, targets, effective_mask)

    loss = ce_alpha * ce_loss + kd_alpha * kd_loss

    # --- Per-position accuracy metrics ---
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    metrics["ce_loss_sum"] = ce_loss.detach().clone()
    metrics["ce_loss_total"] = ones
    metrics["kd_loss_sum"] = kd_loss.detach().clone()
    metrics["kd_loss_total"] = ones
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()

    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
    return loss, metrics


def _kd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    effective_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL divergence from draft to target, weighted by effective_mask."""
    student_log_prob = F.log_softmax(logits.float() / temperature, dim=-1)
    teacher_prob = F.softmax(targets.float() / temperature, dim=-1)
    per_token_kl = F.kl_div(
        student_log_prob, teacher_prob, reduction="none", log_target=False
    ).sum(dim=-1)
    per_token_kl = per_token_kl * (temperature**2)
    denom = effective_mask.sum().clamp_min(1e-5)
    return (per_token_kl * effective_mask).sum() / denom


def _ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    effective_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy against argmax of target logits, weighted by effective_mask."""
    target_ids = torch.argmax(targets, dim=-1)
    batch_size, seq_len, vocab_size = logits.shape
    per_token_ce = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)
    denom = effective_mask.sum().clamp_min(1e-5)
    return (per_token_ce * effective_mask).sum() / denom
