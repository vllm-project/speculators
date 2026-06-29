"""Markov Head implementations for DSpark.

Supports three modes:
- VanillaMarkov: Linear bias from previous token embedding.
- GatedMarkovHead: Gated combination of hidden states and token embedding.
- RNNHead: GRU-like recurrent state across positions within a block.
"""

from typing import Optional

import torch
from torch import nn


class VanillaMarkov(nn.Module):
    """Vanilla Markov head: adds a linear bias based on the previous token.

    bias = W2(W1(prev_token_id))
    logits = base_logits + bias
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for previous token IDs."""
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project latent states to vocabulary bias."""
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute bias for a single step."""
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply Markov bias to single-step logits."""
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply Markov bias to block logits during training.

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids: [B, num_blocks, block_size] previous token IDs.
            hidden_states: [B, num_blocks, block_size, D] (unused for vanilla).

        Returns:
            [B, num_blocks, block_size, V] corrected logits.
        """
        if base_logits.size(2) == 0:
            return base_logits
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias


class GatedMarkovHead(VanillaMarkov):
    """Gated Markov head: uses hidden states to gate the token embedding.

    gate = sigmoid(W_g([hidden_states; W1(prev_token)]))
    bias = W2(gate * W1(prev_token))
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute gating values."""
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute gated bias for a single step."""
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(
            dtype=prev_embeddings.dtype
        )
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):
    """RNN-based Markov head with recurrent state across positions.

    Maintains a GRU-like state that propagates across positions within a block,
    allowing position k to access the full prefix history x_{<k}.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.hidden_size = hidden_size
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(
            2 * markov_rank + hidden_size, 3 * markov_rank
        )

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single RNN step.

        Args:
            state: [*, r] previous recurrent state.
            prev_embeddings: [*, r] W1[x_{k-1}].
            hidden_states: [*, d] backbone hidden at step k.

        Returns:
            new_state: [*, r]
            bias: [*, vocab_size]
        """
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stateless single-step bias (state initialized to zero)."""
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply RNN bias during training (teacher-forced, unrolled over block_size).

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids: [B, num_blocks, block_size] previous token IDs.
            hidden_states: [B, num_blocks, block_size, D]

        Returns:
            [B, num_blocks, block_size, V] corrected logits.
        """
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            output_logits.append(base_logits[..., k, :] + bias)

        return torch.stack(output_logits, dim=-2)


def build_markov_head(config) -> nn.Module | None:
    """Build Markov head from config.

    Args:
        config: DSparkSpeculatorConfig or compatible config with
            markov_rank, markov_head_type, vocab_size, hidden_size.

    Returns:
        Markov head module or None if markov_rank <= 0.
    """
    markov_rank = int(config.markov_rank)
    if markov_rank <= 0:
        return None

    markov_head_type = str(config.markov_head_type).lower()
    vocab_size = config.draft_vocab_size
    hidden_size = config.transformer_layer_config.hidden_size

    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
        )
    raise ValueError(
        f"Unknown markov_head_type: {markov_head_type}. "
        "Expected one of: vanilla, gated, rnn."
    )


__all__ = [
    "VanillaMarkov",
    "GatedMarkovHead",
    "RNNHead",
    "build_markov_head",
]
