"""Markov correction and confidence training over block backbone features."""


import torch

from speculators.losses import LossConfig, kl_div_loss
from speculators.models.dspark.metrics import compute_metrics

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def dspark_objective(
    model,
    hidden_states: torch.Tensor,  # [1, total_seq_len, num_target_layers * hidden_size]
    input_ids: torch.Tensor,  # [1, total_seq_len]
    loss_mask: torch.Tensor,  # [1, total_seq_len]
    verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
    document_ids: torch.Tensor,  # [1, total_seq_len]
    position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
    loss_config: LossConfig | None = None,
    gamma: float = 4.0,
    max_anchors: int = 3072,
    confidence_head_alpha: float = 1.0,
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
    hidden, logits, layout = features
    aligned_loss_mask = layout.loss_mask
    anchored_block_indices = layout.block_positions
    targets = model._build_targets(  # noqa: SLF001
        verifier_last_hidden_states, anchored_block_indices
    )

    # DSpark: add the Markov logit bias and predict per-position confidence.
    num_blocks = max_anchors
    block = model.block_size
    mask_tokens_size = num_blocks * block
    prev_token_ids = layout.predecessor_ids(input_ids)
    hidden_blocks = hidden.view(num_blocks, block, -1)
    corrected = model.correct(features, prev_token_ids)
    logits = corrected.logits
    prev_emb = corrected.confidence_features
    confidence_logits = None

    if model.confidence_head is not None:
        # confidence_head_with_markov requires markov_rank > 0 (enforced in
        # __init__), so prev_emb is always set when the flag is on.
        if model.confidence_with_correction and prev_emb is not None:
            conf_features = torch.cat(
                [hidden_blocks, prev_emb.to(hidden_blocks.dtype)],
                dim=-1,
            )
        else:
            conf_features = hidden_blocks
        confidence_logits = model.confidence_head(conf_features).reshape(
            1, mask_tokens_size
        )

    loss, metrics = compute_metrics(
        logits,
        targets,
        confidence_logits,
        aligned_loss_mask,
        model.block_size,
        loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
        gamma=gamma,
        confidence_head_alpha=confidence_head_alpha,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=model.config.sample_from_anchor,
    )
    return None, loss, metrics  # loss: []
