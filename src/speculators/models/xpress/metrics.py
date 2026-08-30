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
from torch.nn.functional import cross_entropy, softmax

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
)

__all__ = [
    "compute_metrics",
    "greedy_accept_length_metrics",
]


def _hard_label_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    decay_fn,
    denom_decay: bool,
) -> torch.Tensor:
    """CE against HARD data labels with loss_function's weighting semantics
    (mask, optional positional decay, decayed-or-mask denominator). The CE here
    targets the token the sequence actually continued with (the
    target model's own sample at regeneration time), not the teacher argmax."""

    v = logits.shape[-1]
    elem = cross_entropy(
        logits.reshape(-1, v).float(), labels.reshape(-1), reduction="none"
    ).view(1, -1)
    mask = loss_mask.to(elem.dtype)
    elem = elem * mask
    decay_mult = None
    if decay_fn is not None:
        decay_mult = decay_fn(pos_idx.to(elem.dtype), elementwise_loss=elem)
        elem = elem * decay_mult
    if denom_decay and decay_mult is not None:
        denom = (mask * decay_mult).sum(dim=1) + 1e-8
    else:
        denom = mask.sum(dim=1) + 1e-8
    return (elem.sum(dim=1) / denom).mean()


@torch.no_grad()
def greedy_accept_length_metrics(
    refined_tokens: torch.Tensor,
    base_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
    slot_valid: torch.Tensor,
    block_keep: torch.Tensor,
    sample_from_anchor: bool = False,
) -> dict:
    """Greedy accept length of the Jacobi-rolled refiner vs the bare drafter."""
    start = 0 if sample_from_anchor else 1
    tgt = gt_tokens[:, start:]
    valid = slot_valid[:, start:].to(torch.bool)
    keep = block_keep.to(refined_tokens.device).to(torch.float32)
    denom = keep.sum().clamp_min(1.0)

    metrics: dict = {}
    for name, toks in (("refiner", refined_tokens), ("drafter", base_tokens)):
        match = (toks[:, start:] == tgt) & valid
        accept = match.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
        batch_mean = (accept.to(keep.dtype) * keep).sum() / denom
        metrics[f"{name}_accept_len_sum"] = batch_mean
        metrics[f"{name}_accept_len_total"] = torch.ones_like(batch_mean)
    return metrics


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
    decayed_loss_norm: bool = False,
    ce_data_labels: torch.Tensor | None = None,
    data_labels: torch.Tensor | None = None,
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

    ce_labels = ce_data_labels if "ce" in loss_config else None
    if ce_labels is not None:
        _cfg_soft = {k: v for k, v in loss_config.items() if k != "ce"}
        _ce_weight = loss_config["ce"][1]

    def _term_loss(lgts, term_decay_fn):
        """compound loss with the CE component optionally re-pointed at the
        DATA tokens instead of the teacher argmax."""
        if ce_labels is None:
            return compound_loss(
                lgts,
                targets,
                loss_mask,
                pos_idx,
                loss_config=loss_config,
                decay_fn=term_decay_fn,
                denom_decay=decayed_loss_norm,
            )
        soft, terms = (
            compound_loss(
                lgts,
                targets,
                loss_mask,
                pos_idx,
                loss_config=_cfg_soft,
                decay_fn=term_decay_fn,
                denom_decay=decayed_loss_norm,
            )
            if _cfg_soft
            else (lgts.new_zeros((), dtype=torch.float32), {})
        )
        ce = _hard_label_ce(
            lgts, ce_labels, loss_mask, pos_idx, term_decay_fn, decayed_loss_norm
        )
        terms = dict(terms)
        if _cfg_soft and not terms:
            ((_only_name, (_, _only_w)),) = _cfg_soft.items()
            terms[f"{_only_name}_loss"] = (
                soft.detach() / _only_w if _only_w else soft.detach()
            )
        terms["ce_loss"] = ce.detach()
        return soft + _ce_weight * ce, terms

    loss_tf, term_losses = _term_loss(logits_tf, decay_fn)
    loss = loss_tf

    metrics: dict[str, Any] = {}
    ones = torch.ones((), device=device)
    metrics["tf_loss_sum"] = loss_tf.detach().clone()
    metrics["tf_loss_total"] = ones.clone()

    round_losses = []
    for logits_j in round_logits:
        loss_j, _ = _term_loss(logits_j, decay_fn)
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
        loss_base, _ = _term_loss(base_logits, anchor_decay_fn)
        loss = loss + base_anchor_weight * loss_base
        metrics["base_anchor_loss_sum"] = loss_base.detach().clone()
        metrics["base_anchor_loss_total"] = ones.clone()

    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones.clone()
    for term_name, term_val in term_losses.items():
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
        # Top-1 accuracy compares the refined argmax against the DATA token the
        # sequence continued with, not against the teacher's greedy argmax.
        _acc_ref = data_labels if data_labels is not None else targets.argmax(dim=-1)
        acc_hit = (logits_tf.argmax(dim=-1) == _acc_ref).to(mask_f.dtype)
        metrics["accuracy_sum"] = (acc_hit * mask_f).sum()
        metrics["accuracy_total"] = mask_f.sum().clamp_min(1.0)
        if base_logits is not None:
            base_accept = _overlap(base_logits, targets)
            metrics["base_accept_rate_sum"] = (base_accept * mask_f).sum()
            metrics["base_accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    return loss, metrics
