"""Target-free correction operations shared by model recipes and objectives."""

import torch
from torch import nn

from speculators.models.components import (
    CandidateScores,
    CorrectionOutput,
    DraftFeatures,
)


def correct_logits(
    correction: nn.Module,
    features: DraftFeatures,  # hidden/logits: [1, num_blocks * block_size, ...]
    predecessor_ids: torch.Tensor,  # [num_blocks, block_size]
) -> CorrectionOutput:  # logits: [1, num_blocks * block_size, draft_vocab_size]
    """Apply one correction pass with explicitly supplied verifier-token context.

    Predecessors have shape [num_blocks, block_size]. They may be teacher-forced
    or generated: this operation neither reads targets nor chooses that policy.
    """
    if isinstance(correction, nn.Identity):
        return CorrectionOutput(features.logits)
    # Import after model registration; algorithm packages re-export their models.
    from speculators.models.dspark.model_definitions import MarkovHead  # noqa: PLC0415
    from speculators.models.xpress.model_definitions import (  # noqa: PLC0415
        XPressRefinerHead,
    )

    hidden = features.hidden.reshape(*predecessor_ids.shape, -1)
    logits = features.logits.reshape(*predecessor_ids.shape, -1)
    if isinstance(correction, MarkovHead):
        previous = correction.prev_embeddings(predecessor_ids)
        bias = correction.block_bias(
            prev_token_ids=predecessor_ids,
            hidden_states=hidden,
            prev_emb=previous,
        )
        return CorrectionOutput((logits + bias).reshape_as(features.logits), previous)
    if isinstance(correction, XPressRefinerHead):
        cached = correction.hidden_cache(hidden.to(correction.down_h.weight.dtype))
        bias = correction.block_bias(predecessor_ids, hcache=cached)
        return CorrectionOutput((logits + bias).reshape_as(features.logits))
    raise TypeError(
        "This correction requires candidate scoring instead of dense logits"
    )


def score_candidates(
    correction: nn.Module,
    features: DraftFeatures,  # hidden/logits: [1, num_blocks * block_size, ...]
    predecessor_ids: torch.Tensor,  # [num_blocks, block_size]
    candidate_ids: torch.Tensor,  # [1, num_blocks * block_size, top_k]
) -> CandidateScores:  # ids/scores: [1, num_blocks * block_size, top_k]
    """Score caller-selected draft candidates without injecting training targets."""
    from speculators.models.dflash2.model_definitions import (  # noqa: PLC0415
        CandidateSelector,
    )

    if not isinstance(correction, CandidateSelector):
        raise TypeError("Candidate scoring requires a candidate-selector correction")
    return CandidateScores(
        candidate_ids,
        correction.score_candidates(
            features.logits,
            features.hidden,
            predecessor_ids.reshape(1, -1),
            candidate_ids,
        ),
    )
