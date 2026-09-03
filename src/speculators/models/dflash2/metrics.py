"""Runtime-aligned selector loss and metrics for DFlash2."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn import functional

from speculators.losses import (
    LossConfig,
    dflash_loss_decay,
    dpace_loss_decay,
    loss_function,
    tv_loss,
)
from speculators.models.dspark.metrics import compute_metrics as compute_unary_metrics

__all__ = [
    "compute_metrics",
    "compute_selector_loss",
    "selector_confidence_targets",
    "selector_training_candidates",
]


@torch.no_grad()
def selector_confidence_targets(
    *,
    selector_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Return analytical acceptance probability on the serving support."""
    proposal_probs = selector_logits.float().softmax(dim=-1)
    target_log_normalizer = torch.logsumexp(target_logits.float(), dim=-1)
    candidate_target_probs = torch.exp(
        target_logits.float().gather(-1, candidate_ids)
        - target_log_normalizer[..., None]
    )
    return torch.minimum(proposal_probs, candidate_target_probs).sum(dim=-1)


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build top-k selector candidates, injecting a missing target at rank K.

    Unary top-k is the serving candidate set. Training replaces its weakest
    candidate only when the hard target is absent, so every position has a
    well-defined K-way cross-entropy label without expanding the selector to the
    full vocabulary.

    Returns (training_candidate_ids, target_positions, contains_target) where
    contains_target is a boolean mask indicating positions where the target was
    already present in the original unary top-k.
    """
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    target_positions = torch.where(
        contains_target,
        target_positions,
        top_k - 1,
    )

    training_candidate_ids = candidate_ids.clone()
    training_candidate_ids[..., -1] = torch.where(
        contains_target,
        training_candidate_ids[..., -1],
        target_ids,
    )
    return training_candidate_ids, target_positions, contains_target


def _candidate_cross_entropy(
    logits: torch.Tensor,  # [*, top_k]
    target_positions: torch.Tensor,  # [*]
) -> torch.Tensor:
    return functional.cross_entropy(
        logits.flatten(0, -2),
        target_positions.flatten(),
        reduction="none",
    ).view_as(target_positions)


def compute_selector_loss(
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    *,
    gamma: float,
    per_position_loss_weight: str,
    dpace_alpha: float,
    sample_from_anchor: bool = False,
) -> torch.Tensor:
    """Compute teacher-forced hard CE over the runtime-sized candidate set."""
    pos_idx = (
        torch.arange(candidate_logits.shape[1], device=candidate_logits.device)
        % block_size
    ).unsqueeze(0)
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay,
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
        )
    return loss_function(
        candidate_logits,
        target_positions,
        loss_mask,
        pos_idx,
        loss_fn=_candidate_cross_entropy,
        decay_fn=decay_fn,
    )


def compute_metrics(
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    training_candidate_ids: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    contains_target: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    top_k: int,
    sample_from_anchor: bool = False,
    *,
    serving_candidate_ids: torch.Tensor | None = None,
    serving_candidate_logits: torch.Tensor | None = None,
    confidence_logits: torch.Tensor | None = None,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    selector_loss_alpha: float = 1.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine the unary DFlash objective with a K-way selector objective."""
    unary_loss, metrics = compute_unary_metrics(
        unary_logits,
        targets,
        None,
        loss_mask,
        block_size,
        loss_config=loss_config,
        tv_loss_fn=tv_loss_fn,
        gamma=gamma,
        confidence_head_alpha=0.0,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )
    selector_loss = compute_selector_loss(
        candidate_logits,
        target_positions,
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )
    loss = unary_loss + selector_loss_alpha * selector_loss

    one = torch.ones((), device=unary_logits.device)
    metrics["unary_loss_sum"] = unary_loss.detach().clone()
    metrics["unary_loss_total"] = one
    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = one.clone()
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = one.clone()

    if confidence_logits is not None:
        if serving_candidate_ids is None or serving_candidate_logits is None:
            raise ValueError(
                "serving_candidate_ids and serving_candidate_logits are required "
                "when confidence_logits are provided"
            )
        confidence_targets = selector_confidence_targets(
            selector_logits=serving_candidate_logits,
            candidate_ids=serving_candidate_ids,
            target_logits=targets,
        ).float()
        pos_idx = (
            torch.arange(confidence_logits.shape[1], device=confidence_logits.device)
            % block_size
        ).unsqueeze(0)
        if per_position_loss_weight == "dpace":
            decay_fn = partial(
                dpace_loss_decay,
                loss_mask=loss_mask,
                block_size=block_size,
                dpace_alpha=dpace_alpha,
            )
        else:
            decay_fn = partial(
                dflash_loss_decay,
                gamma=gamma,
                sample_from_anchor=sample_from_anchor,
            )
        confidence_loss = loss_function(
            confidence_logits,
            confidence_targets,
            loss_mask,
            pos_idx,
            loss_fn=lambda logits, targets: functional.binary_cross_entropy_with_logits(
                logits.float(), targets.float(), reduction="none"
            ),
            decay_fn=decay_fn,
        )
        loss = loss + confidence_head_alpha * confidence_loss
        valid_float = loss_mask.float()
        valid_total = valid_float.sum().clamp_min(1.0)
        confidence_probs = confidence_logits.float().sigmoid()
        metrics["confidence_loss_sum"] = confidence_loss.detach().clone()
        metrics["confidence_loss_total"] = one.clone()
        metrics["confidence_target_mean_sum"] = (confidence_targets * valid_float).sum()
        metrics["confidence_target_mean_total"] = valid_total
        metrics["confidence_pred_mean_sum"] = (confidence_probs * valid_float).sum()
        metrics["confidence_pred_mean_total"] = valid_total.clone()
        metrics["confidence_abs_error_sum"] = (
            (confidence_probs - confidence_targets).abs() * valid_float
        ).sum()
        metrics["confidence_abs_error_total"] = valid_total.clone()
        start_pos = 0 if sample_from_anchor else 1
        num_blocks = confidence_logits.shape[1] // block_size
        confidence_blocks = confidence_probs.view(num_blocks, block_size)[:, start_pos:]
        target_blocks = confidence_targets.view(num_blocks, block_size)[:, start_pos:]
        draft_mask = loss_mask.float().view(num_blocks, block_size)[:, start_pos:]
        predicted_survival = (confidence_blocks * draft_mask).cumprod(dim=-1)
        target_survival = (target_blocks * draft_mask).cumprod(dim=-1)
        metrics["confidence_cumprod_bias_sum"] = (
            (predicted_survival - target_survival) * draft_mask
        ).sum()
        metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)
        metrics["loss_sum"] = loss.detach().clone()

    with torch.no_grad():
        target_ids = targets.argmax(dim=-1)
        valid = loss_mask.to(torch.bool)
        valid_float = valid.to(unary_logits.dtype)
        valid_total = valid_float.sum()

        metrics[f"unary_candidate_recall_at_{top_k}_sum"] = (
            contains_target.to(valid_float.dtype) * valid_float
        ).sum()
        metrics[f"unary_candidate_recall_at_{top_k}_total"] = valid_total

        target_log_normalizer = torch.logsumexp(targets.float(), dim=-1)
        candidate_target_logits = targets.gather(-1, training_candidate_ids).float()
        candidate_mass = torch.exp(
            torch.logsumexp(candidate_target_logits, dim=-1) - target_log_normalizer
        )
        metrics[f"unary_candidate_target_mass_at_{top_k}_sum"] = (
            candidate_mass * valid_float
        ).sum()
        metrics[f"unary_candidate_target_mass_at_{top_k}_total"] = valid_total.clone()

        teacher_forced_ids = training_candidate_ids.gather(
            -1, candidate_logits.detach().argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        serving_valid = valid_float * contains_target.to(valid_float.dtype)
        serving_total = serving_valid.sum()
        metrics["teacher_forced_selector_acc_sum"] = (
            teacher_forced_ids.eq(target_ids).to(valid_float.dtype) * serving_valid
        ).sum()
        metrics["teacher_forced_selector_acc_total"] = serving_total

        num_blocks = unary_logits.shape[1] // block_size
        contains_target_blocks = contains_target.view(num_blocks, block_size)
        valid_blocks = valid.view(num_blocks, block_size)
        oracle_alive = torch.ones(
            num_blocks, dtype=torch.bool, device=unary_logits.device
        )
        oracle_accepted_length = torch.ones(
            num_blocks, dtype=torch.float32, device=unary_logits.device
        )
        for position in range(1, block_size):
            oracle_alive = (
                oracle_alive
                & valid_blocks[:, position]
                & contains_target_blocks[:, position]
            )
            oracle_accepted_length += oracle_alive.to(oracle_accepted_length.dtype)
        block_valid = valid_blocks[:, 1:].any(dim=-1)
        block_total = block_valid.sum().to(torch.float32)
        metrics[f"unary_top_{top_k}_oracle_accepted_length_sum"] = (
            oracle_accepted_length * block_valid
        ).sum()
        metrics[f"unary_top_{top_k}_oracle_accepted_length_total"] = block_total

    return loss, metrics
