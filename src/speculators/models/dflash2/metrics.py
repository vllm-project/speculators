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
from speculators.models.dflash2.model_definitions import CandidateSelector
from speculators.models.dspark.metrics import compute_metrics as compute_unary_metrics

__all__ = [
    "compute_metrics",
    "compute_selector_loss",
    "selector_training_candidates",
]


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build top-k selector candidates, injecting a missing target at rank K.

    Unary top-k is the serving candidate set. Training replaces its weakest
    candidate only when the hard target is absent, so every position has a
    well-defined K-way cross-entropy label without expanding the selector to the
    full vocabulary.
    """
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    target_positions = torch.where(
        contains_target,
        target_positions,
        torch.full_like(target_positions, top_k - 1),
    )

    training_candidate_ids = candidate_ids.clone()
    training_candidate_ids[..., -1] = torch.where(
        contains_target,
        training_candidate_ids[..., -1],
        target_ids,
    )
    return training_candidate_ids, target_positions


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
    selector: CandidateSelector,
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    hidden_states: torch.Tensor,  # [1, num_anchors*block_size, hidden_size]
    predecessor_ids: torch.Tensor,  # [1, num_anchors*block_size]
    candidate_ids: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_ids: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    *,
    gamma: float,
    per_position_loss_weight: str,
    dpace_alpha: float,
) -> torch.Tensor:
    """Compute teacher-forced hard CE over the runtime-sized candidate set."""
    candidate_ids, target_positions = selector_training_candidates(
        candidate_ids,
        target_ids,
    )
    candidate_logits = selector.score_candidates(
        unary_logits,
        hidden_states,
        predecessor_ids,
        candidate_ids,
    )
    pos_idx = (
        torch.arange(unary_logits.shape[1], device=unary_logits.device) % block_size
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
            sample_from_anchor=False,
        )
    return loss_function(
        candidate_logits,
        target_positions,
        loss_mask,
        pos_idx,
        loss_fn=_candidate_cross_entropy,
        decay_fn=decay_fn,
    )


@torch.no_grad()
def _add_selector_metrics(
    metrics: dict[str, Any],
    *,
    selector: CandidateSelector,
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    hidden_states: torch.Tensor,  # [1, num_anchors*block_size, hidden_size]
    teacher_forced_predecessor_ids: torch.Tensor,  # [1, num_anchors*block_size]
    verified_anchor_ids: torch.Tensor,  # [num_anchors]
    candidate_ids: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_ids: torch.Tensor,  # [1, num_anchors*block_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
) -> None:
    """Add strict-candidate, teacher-forced, and self-conditioned metrics."""
    top_k = selector.top_k
    valid = loss_mask.to(torch.bool)
    valid_float = valid.to(unary_logits.dtype)
    valid_total = valid_float.sum()
    candidate_hit = candidate_ids.eq(target_ids.unsqueeze(-1)).any(dim=-1)

    metrics[f"unary_candidate_recall_at_{top_k}_sum"] = (
        candidate_hit.to(valid_float.dtype) * valid_float
    ).sum()
    metrics[f"unary_candidate_recall_at_{top_k}_total"] = valid_total

    target_log_normalizer = torch.logsumexp(targets.float(), dim=-1)
    candidate_target_logits = targets.gather(-1, candidate_ids).float()
    candidate_mass = torch.exp(
        torch.logsumexp(candidate_target_logits, dim=-1) - target_log_normalizer
    )
    metrics[f"unary_candidate_target_mass_at_{top_k}_sum"] = (
        candidate_mass * valid_float
    ).sum()
    metrics[f"unary_candidate_target_mass_at_{top_k}_total"] = valid_total.clone()

    teacher_forced_scores = selector.score_candidates(
        unary_logits,
        hidden_states,
        teacher_forced_predecessor_ids,
        candidate_ids,
    )
    teacher_forced_ids = candidate_ids.gather(
        -1, teacher_forced_scores.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    metrics["teacher_forced_selector_acc_sum"] = (
        teacher_forced_ids.eq(target_ids).to(valid_float.dtype) * valid_float
    ).sum()
    metrics["teacher_forced_selector_acc_total"] = valid_total.clone()

    num_blocks = unary_logits.shape[1] // block_size
    unary_blocks = unary_logits.view(num_blocks, block_size, -1)
    hidden_blocks = hidden_states.view(num_blocks, block_size, -1)
    candidate_blocks = candidate_ids.view(num_blocks, block_size, top_k)
    target_blocks = target_ids.view(num_blocks, block_size)
    valid_blocks = valid.view(num_blocks, block_size)
    candidate_hit_blocks = candidate_hit.view(num_blocks, block_size)

    previous_ids = verified_anchor_ids
    path_alive = torch.ones(num_blocks, dtype=torch.bool, device=unary_logits.device)
    oracle_alive = path_alive.clone()
    path_accepted_length = torch.ones(
        num_blocks, dtype=torch.float32, device=unary_logits.device
    )
    oracle_accepted_length = path_accepted_length.clone()

    for position in range(1, block_size):
        path_candidate_ids = candidate_blocks[:, position]
        path_scores = selector.score_candidates(
            unary_blocks[:, position],
            hidden_blocks[:, position],
            previous_ids,
            path_candidate_ids,
        )
        selected_ids = path_candidate_ids.gather(
            -1, path_scores.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        position_valid = valid_blocks[:, position]
        path_eligible = path_alive & position_valid
        path_correct = selected_ids.eq(target_blocks[:, position])
        metrics[f"self_conditioned_path_position_{position}_conditional_acc_sum"] = (
            (path_eligible & path_correct).sum().to(torch.float32)
        )
        metrics[f"self_conditioned_path_position_{position}_conditional_acc_total"] = (
            path_eligible.sum().to(torch.float32)
        )

        path_alive = path_eligible & path_correct
        oracle_alive = oracle_alive & position_valid & candidate_hit_blocks[:, position]
        path_accepted_length += path_alive.to(path_accepted_length.dtype)
        oracle_accepted_length += oracle_alive.to(oracle_accepted_length.dtype)
        previous_ids = selected_ids

    block_valid = valid_blocks[:, 1:].any(dim=-1)
    block_total = block_valid.sum().to(torch.float32)
    metrics["self_conditioned_path_accepted_length_sum"] = (
        path_accepted_length * block_valid
    ).sum()
    metrics["self_conditioned_path_accepted_length_total"] = block_total
    metrics[f"unary_top_{top_k}_oracle_accepted_length_sum"] = (
        oracle_accepted_length * block_valid
    ).sum()
    metrics[f"unary_top_{top_k}_oracle_accepted_length_total"] = block_total.clone()


def compute_metrics(
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    hidden_states: torch.Tensor,  # [1, num_anchors*block_size, hidden_size]
    teacher_forced_predecessor_ids: torch.Tensor,  # [1, num_anchors*block_size]
    verified_anchor_ids: torch.Tensor,  # [num_anchors]
    selector: CandidateSelector,
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    *,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    selector_loss_alpha: float = 1.0,
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
        sample_from_anchor=False,
    )
    target_ids = targets.argmax(dim=-1)
    candidate_ids = unary_logits.topk(selector.top_k, dim=-1).indices
    selector_loss = compute_selector_loss(
        selector,
        unary_logits,
        hidden_states,
        teacher_forced_predecessor_ids,
        candidate_ids,
        target_ids,
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
    )
    loss = unary_loss + selector_loss_alpha * selector_loss

    one = torch.ones((), device=unary_logits.device)
    metrics["unary_loss_sum"] = unary_loss.detach().clone()
    metrics["unary_loss_total"] = one
    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = one.clone()
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = one.clone()
    _add_selector_metrics(
        metrics,
        selector=selector,
        unary_logits=unary_logits,
        hidden_states=hidden_states,
        teacher_forced_predecessor_ids=teacher_forced_predecessor_ids,
        verified_anchor_ids=verified_anchor_ids,
        candidate_ids=candidate_ids,
        target_ids=target_ids,
        targets=targets,
        loss_mask=loss_mask,
        block_size=block_size,
    )
    return loss, metrics
