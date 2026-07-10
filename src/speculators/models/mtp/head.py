"""Multi-level LM head for Gemma4 MTP."""

import torch
import torch.nn.functional as F
from torch import nn


class MultiLevelLMHead(nn.Module):
    """Auxiliary centroid projection for Gemma4 MTP."""

    def __init__(self, hidden_size: int, vocab_size: int, num_centroids: int):
        super().__init__()
        self.tokens_per_centroid = vocab_size // num_centroids
        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        # Token ordering mapping (centroid subsets -> token IDs)
        self.register_buffer("token_ordering", torch.arange(vocab_size, dtype=torch.long))

    def compute_centroid_loss(
        self, hidden_states: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """Compute the unreduced centroid cross-entropy loss."""
        valid_mask = targets != ignore_index
        valid_hidden = hidden_states[valid_mask]
        valid_targets = targets[valid_mask]

        loss = torch.zeros_like(targets, dtype=hidden_states.dtype)
        if valid_hidden.shape[0] == 0:
            return loss

        # Inverse mapping: token_id -> centroid_id
        inverse = torch.empty_like(self.token_ordering)
        positions = torch.arange(self.token_ordering.shape[0], device=self.token_ordering.device)
        inverse[self.token_ordering] = positions // self.tokens_per_centroid
        
        centroid_logits = self.centroids(valid_hidden)
        target_centroids = inverse[valid_targets]
        loss[valid_mask] = F.cross_entropy(centroid_logits, target_centroids, reduction="none")
        
        return loss
