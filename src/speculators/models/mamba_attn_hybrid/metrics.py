from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.models.mamba_attn_hybrid.loss import (
    LossConfig,
    ce_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    dpace_loss_weight,
    loss_function,
)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int = 1,
    gamma: float = 4.0,
    parallel_drafting_loss_weight: str = "dflash",
    dpace_alpha: float = 0.5,
    confidence_logits: torch.Tensor | None = None,
    confidence_head_alpha: float = 1.0,
    loss_config: LossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)
    decay_mult = None

    if loss_config is None:
        loss_config = {"ce": (ce_loss, 1.0)}

    ce_is_loss_term = "ce" in loss_config
    if ce_is_loss_term:
        ce_elementwise = ce_loss(logits, targets)
    elif parallel_drafting_loss_weight == "dpace":
        with torch.no_grad():
            ce_elementwise = ce_loss(logits, targets)
    else:
        ce_elementwise = None

    if parallel_drafting_loss_weight == "dflash":
        decay_fn = partial(dflash_loss_decay, gamma=gamma)
    elif parallel_drafting_loss_weight == "dpace":
        decay_mult = dpace_loss_weight(
            ce_elementwise, loss_mask, block_size, alpha=dpace_alpha
        )
        decay_fn = None
    else:
        raise ValueError(
            "Unknown parallel_drafting_loss_weight="
            f"{parallel_drafting_loss_weight!r}; expected 'dflash' or 'dpace'."
        )

    loss = None
    term_losses: dict[str, torch.Tensor] = {}
    multi = len(loss_config) > 1
    for name, (fn, weight) in loss_config.items():
        term_elementwise = ce_elementwise if name == "ce" else fn(logits, targets)
        term = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            decay_fn=decay_fn,
            decay_mult=decay_mult,
            elementwise_loss=term_elementwise,
        )
        if multi:
            term_losses[name] = term.detach()
        contribution = weight * term
        loss = contribution if loss is None else loss + contribution

    conf_loss = None
    if confidence_logits is not None:
        with torch.no_grad():
            draft_p = torch.softmax(logits.float(), dim=-1)
            target_p = torch.softmax(targets.float(), dim=-1)
            accept_target = torch.minimum(draft_p, target_p).sum(dim=-1)
        conf_bce = binary_cross_entropy_with_logits(
            confidence_logits,
            accept_target.to(confidence_logits.dtype),
            reduction="none",
        )
        if decay_mult is not None:
            conf_loss = loss_function(
                logits,
                targets,
                loss_mask,
                pos_idx,
                decay_mult=decay_mult,
                elementwise_loss=conf_bce,
            )
        else:
            conf_loss = loss_function(
                logits,
                targets,
                loss_mask,
                pos_idx,
                elementwise_loss=conf_bce,
                decay_fn=partial(dflash_loss_decay, gamma=gamma),
            )
        loss = loss + confidence_head_alpha * conf_loss

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = torch.tensor(1.0, device=logits.device)
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()

    eal = torch.zeros((), device=logits.device)
    cum = torch.ones((), device=logits.device)
    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
        acc = correct_per_pos[pos] / total_per_pos[pos].clamp(min=1.0)
        cum = cum * acc
        eal = eal + cum
    metrics["eal_sum"] = eal
    metrics["eal_total"] = torch.tensor(1.0, device=logits.device)

    if conf_loss is not None:
        metrics["confidence_loss_sum"] = conf_loss.detach().clone()
        metrics["confidence_loss_total"] = torch.tensor(1.0, device=logits.device)
    for name, term_val in term_losses.items():
        metrics[f"{name}_loss_sum"] = term_val
        metrics[f"{name}_loss_total"] = torch.tensor(1.0, device=logits.device)
    return loss, metrics
