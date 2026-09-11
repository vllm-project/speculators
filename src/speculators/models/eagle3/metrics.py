"""Metrics and loss functions for Eagle3 draft model."""

from functools import partial

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    exp_loss_decay,
    kl_div_loss,
)


def align_for_step(
    logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    ttt_step: int,
):
    """Align logits, targets, and loss_mask for a given ttt_step.

    There are no target values for the last ttt_step tokens, so we mask them out
    before computing the loss. Likewise, there are no logits for the first
    ttt_step tokens, so we mask them out.
    This is equivalent to shifting the target values by ttt_step + 1 to the left
    which puts them in the correct position for the generated tokens.
    e.g.
        indices of targets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices of logits for ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9] # no shift
        indices of logits for ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10] # shift by 1
        indices of logits for ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11] # shift by 2
    The indices for the loss_mask need to be kept in line with the targets indices
    """
    logits = logits[:, :-ttt_step] if ttt_step > 0 else logits
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets = targets[:, ttt_step:]
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    if loss_mask is not None:
        loss_mask = loss_mask[:, ttt_step:]
        # shape: [1, total_seq_len - ttt_step]
    return logits, targets, loss_mask


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    ttt_step: int,
    ttt_step_loss_decay: float,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given ttt_step.

    Args:
        logits: The logits for the current ttt_step.
        targets: The targets for the current ttt_step.
        loss_mask: The loss mask for the current ttt_step.
        ttt_step: The current ttt_step.
        ttt_step_loss_decay: The loss decay for the current ttt_step.
        loss_config: Mapping of ``{name: (loss_fn, weight)}``.

    Returns:
        Loss value and metrics dictionary.
    """
    if loss_config is None:
        loss_config = {"kl_div": (kl_div_loss, 1.0)}
    s_logits, s_targets, s_loss_mask = align_for_step(
        logits, targets, loss_mask, ttt_step
    )

    seq_len = s_logits.shape[1]
    if s_loss_mask is None:
        s_loss_mask = torch.ones(1, seq_len, device=s_logits.device, dtype=torch.bool)

    pos_idx = torch.full(
        (1, seq_len), ttt_step, device=s_logits.device, dtype=torch.long
    )

    s_loss, term_losses = compound_loss(
        s_logits,
        s_targets,
        s_loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=partial(exp_loss_decay, gamma=ttt_step_loss_decay),
    )

    ones = torch.tensor(1.0, device=s_loss.device)
    s_metrics = {}
    s_metrics[f"loss_{ttt_step}_sum"] = s_loss.detach().clone()
    s_metrics[f"loss_{ttt_step}_total"] = ones
    for term_name, term_val in term_losses.items():
        s_metrics[f"{term_name}_{ttt_step}_sum"] = term_val
        s_metrics[f"{term_name}_{ttt_step}_total"] = ones.clone()
    return s_loss, s_metrics
