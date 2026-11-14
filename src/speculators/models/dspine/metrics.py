"""Training objective and diagnostics for DSpine."""

from typing import Any

import torch
from torch.nn import functional

from speculators.losses import dflash_loss_decay
from speculators.losses.eager import ce_loss
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

__all__ = [
    "compute_dspine_loss",
]


def _weighted_mean(
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    position_weights: torch.Tensor,
) -> torch.Tensor:
    weights = loss_mask.to(values.dtype) * position_weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_dspine_loss(
    *,
    unary_logits: torch.Tensor,
    refined_candidate_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    targets: torch.Tensor,
    target_ids: torch.Tensor,
    layer_features: list[torch.Tensor],
    target_features: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    gamma: float,
    alpha: float,
    embedding_beta: float,
    embedding_warmup_steps: int,
    global_step: int,
    transition_min_target_mass: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DSpine losses with verifier-argmax CE and teacher distributions."""
    seq_len = unary_logits.shape[1]
    positions = torch.arange(seq_len, device=unary_logits.device) % block_size
    positions = positions.unsqueeze(0)
    position_weights = dflash_loss_decay(
        positions.to(torch.float32), gamma=gamma, sample_from_anchor=False
    )

    target_probs = functional.softmax(targets.float(), dim=-1)
    ce_target_ids = targets.argmax(dim=-1)
    unary_ce_values = ce_loss(unary_logits.float(), targets)
    unary_probs = functional.softmax(unary_logits.float(), dim=-1)
    unary_l1_values = (unary_probs - target_probs).abs().sum(dim=-1)
    unary_ce = _weighted_mean(unary_ce_values, loss_mask, position_weights)
    unary_l1 = _weighted_mean(unary_l1_values, loss_mask, position_weights)
    unary_loss = alpha * unary_ce + (1.0 - alpha) * unary_l1

    candidate_target_probs = target_probs.gather(-1, candidate_ids)
    candidate_mass = candidate_target_probs.sum(dim=-1)
    renormalized_targets = candidate_target_probs / candidate_mass.unsqueeze(
        -1
    ).clamp_min(1e-12)
    candidate_log_probs = functional.log_softmax(
        refined_candidate_logits.float(), dim=-1
    )
    ce_target_matches = candidate_ids.eq(ce_target_ids.unsqueeze(-1))
    contains_ce_target = ce_target_matches.any(dim=-1)
    # Preserve the global verifier argmax when tied tokens reorder in the pool.
    refined_ce_values = ce_loss(
        refined_candidate_logits.float(), ce_target_matches.float()
    )
    refined_l1_values = (
        (candidate_log_probs.exp() - renormalized_targets).abs().sum(dim=-1)
    )

    refined_ce_mask = loss_mask * contains_ce_target.to(loss_mask.dtype)
    refined_l1_mask = loss_mask * candidate_mass.gt(transition_min_target_mass).to(
        loss_mask.dtype
    )
    refined_ce = _weighted_mean(refined_ce_values, refined_ce_mask, position_weights)
    refined_l1 = _weighted_mean(refined_l1_values, refined_l1_mask, position_weights)
    refined_loss = alpha * refined_ce + (1.0 - alpha) * refined_l1
    distribution_loss = unary_loss + refined_loss

    embedding_loss = unary_logits.new_zeros((), dtype=torch.float32)
    for features in layer_features:
        cosine = functional.cosine_similarity(
            features.float(), target_features.float(), dim=-1
        )
        embedding_loss = embedding_loss + _weighted_mean(
            1.0 - cosine, loss_mask, position_weights
        )

    if embedding_warmup_steps == 0:
        beta = embedding_beta
    else:
        beta = embedding_beta * min(global_step / embedding_warmup_steps, 1.0)
    embedding_lambda = unary_ce.detach() * beta
    loss = distribution_loss + embedding_lambda * embedding_loss

    one = unary_logits.new_ones(())
    metrics: dict[str, Any] = {}
    for name, value in {
        "loss": loss,
        "distribution_loss": distribution_loss,
        "unary_ce": unary_ce,
        "unary_l1": unary_l1,
        "refined_ce": refined_ce,
        "refined_l1": refined_l1,
        "embedding_loss": embedding_loss,
        "embedding_lambda": embedding_lambda,
    }.items():
        metrics[f"{name}_sum"] = value.detach().clone()
        metrics[f"{name}_total"] = one.clone()

    with torch.no_grad():
        # Keep diagnostics referenced to the actual training tokens.
        contains_target = candidate_ids.eq(target_ids.unsqueeze(-1)).any(dim=-1)
        valid = loss_mask.to(torch.bool)
        valid_float = valid.float()
        metrics[f"unary_candidate_recall_at_{candidate_ids.shape[-1]}_sum"] = (
            contains_target.to(valid_float.dtype) * valid_float
        ).sum()
        metrics[f"unary_candidate_recall_at_{candidate_ids.shape[-1]}_total"] = (
            valid_float.sum()
        )

        selected_ids = candidate_ids.gather(
            -1, refined_candidate_logits.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        correct = selected_ids.eq(target_ids)
        start_pos = 1
        # Match DFlash2/DSpark's token accuracy metric names and shared helper.
        correct_per_pos, total_per_pos = compute_accuracy_multi_step(
            selected_ids, target_ids, loss_mask, positions, block_size
        )
        metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
        metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
        for pos in range(start_pos, block_size):
            metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
            metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

        num_blocks = seq_len // block_size
        draft_correct = correct.view(num_blocks, block_size)[:, start_pos:]
        draft_valid = valid.view(num_blocks, block_size)[:, start_pos:]
        eal_sum, eal_total = compute_accepted_length_counts(draft_correct, draft_valid)
        metrics["eal_sum"] = eal_sum
        metrics["eal_total"] = eal_total

        # Teacher-forced prefix acceptance, excluding the known anchor.
        # Position 1 is the first draft token (paper p. 3, Section 2.1).
        accepted_prefix = (draft_correct & draft_valid).to(torch.float32).cumprod(-1)
        accepted_per_position = accepted_prefix.sum(dim=0)
        for position in range(block_size - start_pos):
            # Use all valid blocks as denominator, including an earlier rejection
            # or a masked suffix. This keeps sum(position rates) == EAL - 1.
            metrics[f"acceptance_rate_pos_{position + 1}_sum"] = accepted_per_position[
                position
            ].clone()
            metrics[f"acceptance_rate_pos_{position + 1}_total"] = eal_total.clone()

    return loss, metrics
