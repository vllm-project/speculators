"""Shared model accuracy and acceptance metrics."""

import torch


@torch.no_grad()
def compute_greedy_block_metrics(
    correct: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool,
) -> dict[str, torch.Tensor]:
    """Pool greedy accuracy and accepted-length counts across draft blocks."""
    start_pos = 0 if sample_from_anchor else 1
    correct_slots = correct.reshape(-1, block_size)[:, start_pos:]
    valid_slots = loss_mask.to(torch.bool).reshape(-1, block_size)[:, start_pos:]
    correct_and_valid = correct_slots & valid_slots
    correct_per_pos = correct_and_valid.sum(dim=0, dtype=torch.float32)
    total_per_pos = valid_slots.sum(dim=0, dtype=torch.float32)
    # Count only the leading run of correct slots, plus the verifier bonus token.
    accepted_length = (
        correct_and_valid.cumprod(dim=-1, dtype=torch.float32).sum(dim=-1) + 1.0
    )
    valid_blocks = valid_slots.any(dim=-1).to(torch.float32)

    metrics = {
        "full_acc_sum": correct_per_pos.sum(),
        "full_acc_total": total_per_pos.sum(),
        "eal_sum": (accepted_length * valid_blocks).sum(),
        "eal_total": valid_blocks.sum(),
    }
    for offset, position in enumerate(range(start_pos, block_size)):
        metrics[f"position_{position}_acc_sum"] = correct_per_pos[offset]
        metrics[f"position_{position}_acc_total"] = total_per_pos[offset]
    return metrics


@torch.no_grad()
def compute_acceptance_rate_metrics(
    probabilities: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    name: str = "accept_rate",
) -> dict[str, torch.Tensor]:
    """Aggregate per-token acceptance probabilities over the loss mask."""
    mask = loss_mask.to(probabilities.dtype)
    return {
        f"{name}_sum": (probabilities * mask).sum(),
        f"{name}_total": mask.sum(),
    }


@torch.no_grad()
def compute_analytical_block_acceptance(
    probabilities: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool,
    *,
    prefix: str = "",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Report analytical acceptance and return the prefixes and mask for calibration."""
    start_pos = 0 if sample_from_anchor else 1
    draft_slots = probabilities.reshape(-1, block_size)[:, start_pos:]
    draft_mask = loss_mask.to(probabilities.dtype).reshape(-1, block_size)[
        :, start_pos:
    ]
    accept_prefix = (draft_slots * draft_mask).cumprod(dim=-1)
    accepted_length = accept_prefix.sum(dim=-1) + 1.0  # Verifier bonus token.
    valid_blocks = (draft_mask.sum(dim=-1) > 0).to(probabilities.dtype)
    metrics = compute_acceptance_rate_metrics(
        probabilities, loss_mask, name=f"{prefix}accept_rate"
    )
    metrics[f"{prefix}accept_len_sum"] = (accepted_length * valid_blocks).sum()
    metrics[f"{prefix}accept_len_total"] = valid_blocks.sum()
    return metrics, accept_prefix, draft_mask


def compute_accuracy_single_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor | None,  # shape: [1, seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, seq_len]
):
    """Compute full and conditional accuracy counts for a single speculative step.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: If provided, restricts accuracy to masked positions.
        prev_correct: Boolean mask of positions correct so far. Updated in place
            via logical AND with the current step's correctness.

    Returns:
        Tuple of (full_correct, full_total, cond_correct, cond_total) as raw
        counts suitable for distributed reduction before computing ratios.
    """
    correct = pred_ids == target_ids
    numel = torch.full((), correct.numel(), dtype=torch.float, device=correct.device)
    cond_total = numel
    if prev_correct is not None:
        cond_total = prev_correct.sum().float()
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        mask = loss_mask.to(torch.bool)
        correct_sum = torch.logical_and(correct, mask).float().sum()
        full_total = mask.sum().float()
    else:
        correct_sum = correct.float().sum()
        full_total = numel.clone()

    return correct_sum, full_total, correct_sum.clone(), cond_total


@torch.no_grad()
def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    num_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position correct/total counts across multiple speculative steps.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: Boolean mask selecting positions to evaluate.
        pos_idx: Position index within each speculative block (e.g. 0,1,2,3,0,1,2,3).
        num_pos: Number of distinct positions (i.e. block size).

    Returns:
        Tuple of (correct_per_pos, total_per_pos) both with shape [num_pos].
        Overall counts can be derived by summing these.
    """
    correct = pred_ids == target_ids
    correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    pos_idx = torch.masked_select(pos_idx, loss_mask.to(torch.bool))

    correct_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    total_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    correct_per_pos.scatter_add_(0, pos_idx, correct.float())
    total_per_pos.scatter_add_(0, pos_idx, torch.ones_like(correct, dtype=torch.float))

    return correct_per_pos, total_per_pos  # shape: [num_pos], [num_pos]
