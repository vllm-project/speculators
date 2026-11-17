"""Candidate reranking training strategy over prepared backbone features."""

from typing import Any

import torch

from speculators.losses import LossConfig, kl_div_loss
from speculators.models.dflash2.metrics import (
    compute_metrics,
    selector_training_candidates,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def dflash2_objective(
    model,
    hidden_states: torch.Tensor,  # [1, total_seq_len, num_target_layers * hidden_size]
    input_ids: torch.Tensor,  # [1, total_seq_len]
    loss_mask: torch.Tensor,  # [1, total_seq_len]
    verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
    document_ids: torch.Tensor,  # [1, total_seq_len]
    position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
    loss_config: LossConfig | None = None,
    gamma: float = 4.0,
    max_anchors: int = 512,
    selector_loss_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    selector_miss_policy: str = "replace",
    **kwargs,
) -> tuple[None, torch.Tensor, dict[str, Any]]:
    features = model.draft_features(
        hidden_states,
        input_ids,
        loss_mask,
        document_ids,
        position_ids,
        max_anchors=max_anchors,
        **kwargs,
    )
    hidden, unary_logits, layout = features
    aligned_loss_mask = layout.loss_mask
    block_indices = layout.block_positions
    targets = model._build_targets(  # noqa: SLF001
        verifier_last_hidden_states, block_indices
    )
    predecessor_ids = layout.predecessor_ids(input_ids)

    target_ids = targets.argmax(dim=-1)
    # shape: [1, num_anchors*block_size]
    candidate_ids = unary_logits.topk(model.correction.top_k, dim=-1).indices
    # shape: [1, num_anchors*block_size, top_k]
    training_candidate_ids, target_positions, contains_target = (
        selector_training_candidates(
            candidate_ids, target_ids, miss_policy=selector_miss_policy
        )
    )
    candidate_logits = model.correction.score_candidates(
        unary_logits,
        hidden,
        predecessor_ids.reshape(1, -1),
        training_candidate_ids,
    )

    loss, metrics = compute_metrics(
        unary_logits=unary_logits,
        targets=targets,
        training_candidate_ids=training_candidate_ids,
        candidate_logits=candidate_logits,
        target_positions=target_positions,
        contains_target=contains_target,
        loss_mask=aligned_loss_mask,
        block_size=model.block_size,
        top_k=model.correction.top_k,
        sample_from_anchor=model.config.sample_from_anchor,
        loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
        gamma=gamma,
        selector_loss_alpha=selector_loss_alpha,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        selector_miss_policy=selector_miss_policy,
    )
    return None, loss, metrics  # loss: []
