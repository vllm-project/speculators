"""Runtime-aligned selector loss and metrics for DFlash2.

``eal`` follows the realized greedy selector path. ``unary_accept_len`` remains the
analytical TV-overlap estimate over the unary logits.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch.nn import functional

from speculators.losses import (
    LossConfig,
    compound_loss,
    loss_function,
)
from speculators.models.dflash.utils import block_positions_and_loss_decay
from speculators.models.metrics import (
    compute_analytical_block_acceptance,
    compute_base_metrics,
)

__all__ = [
    "compute_metrics",
    "compute_selector_loss",
    "selector_training_candidates",
]


class SelectorTrainingOutput(NamedTuple):
    unary_candidate_ids: torch.Tensor
    candidate_logits: torch.Tensor
    target_positions: torch.Tensor
    contains_target: torch.Tensor
    selected_token_ids: torch.Tensor
    target_ids: torch.Tensor


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
    miss_policy: str = "replace",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build selector candidates under the configured gold-miss policy."""
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)

    if miss_policy == "replace":
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
    elif miss_policy == "strict":
        # Do not inject gold targets. Miss positions receive a dummy label,
        # whose CE is masked in compute_selector_loss().
        training_candidate_ids = candidate_ids
        target_positions = torch.where(
            contains_target,
            target_positions,
            torch.zeros_like(target_positions),
        )
    else:
        raise ValueError(
            f"Unknown selector miss policy: {miss_policy!r}; "
            "expected 'replace' or 'strict'."
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
    selector_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute selector CE, optionally excluding positions with a top-K miss.

    The original DFlash ``loss_mask`` remains the reduction denominator. In
    strict mode, ``selector_valid_mask`` masks selector-loss positions only.
    """
    pos_idx, decay_fn = block_positions_and_loss_decay(
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )

    return _compute_weighted_selector_loss(
        candidate_logits,
        target_positions,
        loss_mask,
        pos_idx,
        decay_fn,
        selector_valid_mask,
    )


def _compute_weighted_selector_loss(
    candidate_logits: torch.Tensor,
    target_positions: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    decay_fn: Callable[..., torch.Tensor],
    selector_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if selector_valid_mask is None:
        selector_loss_mask = loss_mask.to(torch.bool)
    else:
        selector_loss_mask = loss_mask.to(torch.bool) & selector_valid_mask.to(
            torch.bool
        )

    def selector_ce(
        logits: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        ce = _candidate_cross_entropy(logits, positions)
        return ce * selector_loss_mask.to(ce.dtype)

    return loss_function(
        candidate_logits,
        target_positions,
        loss_mask,
        pos_idx,
        loss_fn=selector_ce,
        decay_fn=decay_fn,
    )


def _compute_candidate_target_mass(
    targets: torch.Tensor,
    candidate_ids: torch.Tensor,
    valid: torch.Tensor,
    valid_total: torch.Tensor,
    top_k: int,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    # Process rows in chunks so logsumexp does not materialize a
    # full-vocab fp32 copy of the targets and its exp workspace.
    target_log_normalizer = torch.empty(
        targets.shape[:-1], device=targets.device, dtype=torch.float32
    )
    for start in range(0, targets.shape[1], 1024):
        torch.logsumexp(
            targets[:, start : start + 1024].float(),
            dim=-1,
            out=target_log_normalizer[:, start : start + 1024],
        )
    candidate_target_logits = targets.gather(-1, candidate_ids).float()
    candidate_mass = torch.exp(
        torch.logsumexp(candidate_target_logits, dim=-1) - target_log_normalizer
    )
    metrics[f"unary_candidate_target_mass_at_{top_k}_sum"] = (
        candidate_mass * valid
    ).sum(dtype=torch.float32)
    metrics[f"unary_candidate_target_mass_at_{top_k}_total"] = valid_total.clone()
    return metrics


def _compute_top_k_oracle_accepted_length(
    contains_target: torch.Tensor,
    valid: torch.Tensor,
    block_size: int,
    top_k: int,
    sample_from_anchor: bool,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    num_blocks = valid.shape[1] // block_size
    contains_target_blocks = contains_target.view(num_blocks, block_size)
    valid_blocks = valid.view(num_blocks, block_size)

    oracle_alive = torch.ones(num_blocks, dtype=torch.bool, device=valid.device)
    oracle_accepted_length = torch.ones(
        num_blocks, dtype=torch.float32, device=valid.device
    )
    start_pos = 0 if sample_from_anchor else 1
    for position in range(start_pos, block_size):
        oracle_alive = (
            oracle_alive
            & valid_blocks[:, position]
            & contains_target_blocks[:, position]
        )
        oracle_accepted_length += oracle_alive.to(oracle_accepted_length.dtype)

    block_valid = valid_blocks[:, start_pos:].any(dim=-1)
    block_total = block_valid.sum().to(torch.float32)
    metrics[f"unary_top_{top_k}_oracle_accepted_length_sum"] = (
        oracle_accepted_length * block_valid
    ).sum()
    metrics[f"unary_top_{top_k}_oracle_accepted_length_total"] = block_total
    return metrics


def compute_metrics(
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    correction: SelectorTrainingOutput,
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    top_k: int,
    sample_from_anchor: bool = False,
    *,
    loss_config: LossConfig,
    gamma: float = 4.0,
    selector_loss_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    selector_miss_policy: str = "replace",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine the unary DFlash objective with the selector objective."""
    if selector_miss_policy not in {"replace", "strict"}:
        raise ValueError(
            f"Unknown selector miss policy: {selector_miss_policy!r}; "
            "expected 'replace' or 'strict'."
        )

    pos_idx, decay_fn = block_positions_and_loss_decay(
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )
    unary = compound_loss(
        unary_logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        metric_terms=("tv",),
    )
    unary_loss = unary.loss
    metrics, _, _ = compute_analytical_block_acceptance(
        1.0 - unary.metric_terms["tv"],
        loss_mask,
        block_size,
        sample_from_anchor,
        prefix="unary_",
    )

    selector_valid_mask = (
        correction.contains_target if selector_miss_policy == "strict" else None
    )
    selector_loss = _compute_weighted_selector_loss(
        correction.candidate_logits,
        correction.target_positions,
        loss_mask,
        pos_idx,
        decay_fn,
        selector_valid_mask,
    )
    loss = unary_loss + selector_loss_alpha * selector_loss

    one = torch.ones((), device=unary_logits.device)
    metrics["unary_loss_sum"] = unary_loss.detach().clone()
    metrics["unary_loss_total"] = one
    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = one.clone()

    with torch.no_grad():
        valid = loss_mask.to(torch.bool)
        valid_total = valid.sum(dtype=torch.float32)

        selected_matches_target = correction.selected_token_ids.eq(
            correction.target_ids
        )
        # Gate on the original unary candidate set: selector training must not
        # make injected gold tokens appear available at inference.
        selector_correct = selected_matches_target & correction.contains_target
        metrics.update(
            compute_base_metrics(
                loss,
                unary.term_losses,
                selector_correct,
                valid,
                block_size,
                sample_from_anchor,
            )
        )

        recall_hits = valid & correction.contains_target
        recall_sum = recall_hits.sum(dtype=torch.float32)
        metrics[f"unary_candidate_recall_at_{top_k}_sum"] = recall_sum
        metrics[f"unary_candidate_recall_at_{top_k}_total"] = valid_total
        metrics.update(
            _compute_candidate_target_mass(
                targets, correction.unary_candidate_ids, valid, valid_total, top_k
            )
        )
        metrics["teacher_forced_selector_acc_sum"] = (valid & selector_correct).sum(
            dtype=torch.float32
        )
        metrics["teacher_forced_selector_acc_total"] = recall_sum
        metrics.update(
            _compute_top_k_oracle_accepted_length(
                correction.contains_target, valid, block_size, top_k, sample_from_anchor
            )
        )

    return loss, metrics
