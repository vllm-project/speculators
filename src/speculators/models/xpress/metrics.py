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

from functools import partial
from typing import Any

import torch
from torch.nn.functional import softmax

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
)
from speculators.models.metrics import compute_greedy_block_metrics

__all__ = [
    "compute_metrics",
]


def compute_metrics(  # noqa: C901
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
    seq_len = logits_tf.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    def _term_loss(logits, term_decay_fn):
        return compound_loss(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_config=loss_config,
            decay_fn=term_decay_fn,
        )

    teacher_forced = _term_loss(logits_tf, decay_fn)
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
        loss_base = _term_loss(base_logits, anchor_decay_fn).loss
        loss = loss + base_anchor_weight * loss_base
        metrics["base_anchor_loss_sum"] = loss_base.detach().clone()
        metrics["base_anchor_loss_total"] = ones.clone()

    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones.clone()
    for term_name, term_val in teacher_forced.term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Analytical per-position acceptance rate of the (teacher-forced) refined
    # drafter = distributional overlap with the target.
    with torch.no_grad():

        def _overlap(a_logits, b_logits, chunk=1024):
            outs = []
            for i in range(0, a_logits.shape[1], chunk):
                ap = softmax(a_logits[:, i : i + chunk].float(), dim=-1)
                bp = softmax(b_logits[:, i : i + chunk].float(), dim=-1)
                outs.append(torch.minimum(ap, bp).sum(dim=-1))
            return torch.cat(outs, dim=1)  # [1, T]

        accept_rate = _overlap(logits_tf, targets)
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)
        pred_ids = logits_tf.argmax(dim=-1)
        target_ids = targets.argmax(dim=-1)
        metrics.update(
            compute_greedy_block_metrics(
                pred_ids == target_ids,
                loss_mask,
                block_size,
                sample_from_anchor,
                eal_name="teacher_forced_eal",
            )
        )
        if base_logits is not None:
            base_accept = _overlap(base_logits, targets)
            metrics["base_accept_rate_sum"] = (base_accept * mask_f).sum()
            metrics["base_accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    return loss, metrics
