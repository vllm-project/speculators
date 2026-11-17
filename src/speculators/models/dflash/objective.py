"""Anchored-block distillation, independent of backbone construction."""

import torch

from speculators.losses import LossConfig
from speculators.models.dflash.metrics import compute_metrics


def dflash_objective(
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
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    **kwargs,
):
    features = model.draft_features(
        hidden_states,
        input_ids,
        loss_mask,
        document_ids,
        position_ids,
        max_anchors=max_anchors,
        **kwargs,
    )
    logits = features.logits
    aligned_loss_mask = features.layout.loss_mask
    targets = model._build_targets(  # noqa: SLF001
        verifier_last_hidden_states, features.layout.block_positions
    )
    loss, metrics = compute_metrics(
        logits,
        targets,
        aligned_loss_mask,
        model.block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=model.config.sample_from_anchor,
    )
    return None, loss, metrics  # loss: []
