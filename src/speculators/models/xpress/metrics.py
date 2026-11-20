"""Loss and metrics for the XPress draft model.

loss = compound_loss(refined_tf)                       # teacher-forced prev
     + consistency_weight * mean_j compound_loss(refined_round_j)
     + base_anchor_weight * compound_loss(base_logits)  # drafter anchor

The consistency rounds re-feed the refiner its own argmax prev (round 0 seeds
from the drafter's parallel argmax) — the input distribution the head actually
meets during inference-time Jacobi decoding. Round losses are AVERAGED so
``consistency_weight`` keeps its meaning as the round count varies. The anchor
term holds the co-trained backbone's own logits to the target so the Jacobi
seed does not decay: without it the backbone drifts to serve the refiner and the
standalone drafter (the K=0 seed) degrades over a long co-training run.
"""

from typing import Any

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
)
from speculators.models.dflash.utils import block_positions_and_loss_decay
from speculators.models.metrics import (
    compute_acceptance_rate_metrics,
    compute_greedy_block_metrics,
)

__all__ = [
    "compute_metrics",
]


def compute_metrics(
    logits_tf: torch.Tensor,
    round_logits: list[torch.Tensor],
    base_logits: torch.Tensor | None,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    consistency_weight: float = 0.3,
    base_anchor_weight: float | torch.Tensor = 0.6,
    base_anchor_full_weight: bool = False,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute the XPress loss and a metrics dict (``*_sum``/``*_total`` pairs)."""
    device = logits_tf.device
    pos_idx, decay_fn = block_positions_and_loss_decay(
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )

    def _term_loss(logits, term_decay_fn, *, with_acceptance=False, with_ids=False):
        return compound_loss(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_config=loss_config,
            decay_fn=term_decay_fn,
            metric_terms=("tv",) if with_acceptance else (),
            with_ids=with_ids,
        )

    teacher_forced = _term_loss(
        logits_tf, decay_fn, with_acceptance=True, with_ids=True
    )
    loss_tf = teacher_forced.loss
    loss = loss_tf

    metrics: dict[str, Any] = {}
    ones = torch.ones((), device=device)
    metrics["tf_loss_sum"] = loss_tf.detach().clone()
    metrics["tf_loss_total"] = ones.clone()

    round_losses = []
    for logits_j in round_logits:
        loss_j = _term_loss(logits_j, decay_fn).loss
        round_losses.append(loss_j)
    if round_losses:
        loss_cons = torch.stack(round_losses).mean()
        loss = loss + consistency_weight * loss_cons
        metrics["consistency_loss_sum"] = loss_cons.detach().clone()
        metrics["consistency_loss_total"] = ones.clone()
        for j, loss_j in enumerate(round_losses):
            metrics[f"consistency_round{j}_loss_sum"] = loss_j.detach().clone()
            metrics[f"consistency_round{j}_loss_total"] = ones.clone()

    if base_logits is not None:
        anchor_decay_fn = None if base_anchor_full_weight else decay_fn
        base = _term_loss(base_logits, anchor_decay_fn, with_acceptance=True)
        loss_base = base.loss
        metrics.update(
            compute_acceptance_rate_metrics(
                1.0 - base.metric_terms["tv"], loss_mask, name="base_accept_rate"
            )
        )
        loss = loss + base_anchor_weight * loss_base
        metrics["base_anchor_loss_sum"] = loss_base.detach().clone()
        metrics["base_anchor_loss_total"] = ones.clone()

    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones.clone()
    for term_name, term_val in teacher_forced.term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    metrics.update(
        compute_acceptance_rate_metrics(
            1.0 - teacher_forced.metric_terms["tv"], loss_mask
        )
    )
    metrics.update(
        compute_greedy_block_metrics(
            teacher_forced.pred_ids == teacher_forced.target_ids,
            loss_mask,
            block_size,
            sample_from_anchor,
            eal_name="teacher_forced_eal",
        )
    )

    return loss, metrics
