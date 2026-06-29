"""Confidence Head for DSpark.

Predicts the acceptance probability of each draft position,
enabling early-stop during speculative decoding inference.
"""

import torch
from torch import nn


class AcceptRatePredictor(nn.Module):
    """Predicts the acceptance rate for each draft position.

    Maps hidden state features to a scalar logit representing
    the predicted probability that the verifier will accept this token.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict acceptance logits.

        Args:
            features: [..., input_dim] hidden state features.

        Returns:
            [...] acceptance logits (before sigmoid).
        """
        return self.proj(features).squeeze(-1)


def build_confidence_head(config) -> nn.Module | None:
    """Build confidence head from config.

    Args:
        config: DSparkSpeculatorConfig or compatible config with
            enable_confidence_head, confidence_head_with_markov,
            hidden_size, markov_rank.

    Returns:
        AcceptRatePredictor or None if confidence head is disabled.
    """
    if not config.enable_confidence_head:
        return None

    input_dim = config.transformer_layer_config.hidden_size
    if config.confidence_head_with_markov:
        input_dim += config.markov_rank

    return AcceptRatePredictor(input_dim=input_dim)


__all__ = [
    "AcceptRatePredictor",
    "build_confidence_head",
]
