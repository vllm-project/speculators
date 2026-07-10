"""Local head modules for the DeLS-Spec draft model."""

import torch
from torch import nn

__all__ = [
    "MarkovLocalHead",
    "RNNLocalHead",
]


class RNNLocalHead(nn.Module):
    """GRU-based local head that captures intra-block causal dependencies.

    Processes the block prefix sequentially through a bias-free GRU, projects
    via a low-rank bottleneck with SiLU activation, and outputs full-vocabulary
    logits.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        gru_hidden_size: int,
        low_rank_dim: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        self.w_rank = nn.Linear(gru_hidden_size, low_rank_dim, bias=False)
        self.w_vocab = nn.Linear(low_rank_dim, vocab_size, bias=False)
        self.act = nn.SiLU()

    def forward(
        self,
        embeddings: torch.Tensor,  # [num_blocks, block_size, embed_dim]
    ) -> torch.Tensor:  # [num_blocks, block_size, vocab_size]
        """Run the GRU over block token embeddings and produce per-position logits.

        The GRU output at position j (having seen tokens 0..j) predicts the
        token at position j+1.  The caller is responsible for aligning logits
        with the correct targets.
        """
        gru_out, _ = self.gru(embeddings)  # [num_blocks, block_size, gru_hidden]
        rank_out = self.act(self.w_rank(gru_out))  # [num_blocks, block_size, rank]
        return self.w_vocab(rank_out)  # [num_blocks, block_size, vocab_size]


class MarkovLocalHead(nn.Module):
    """First-order Markov (bigram) local head.

    Uses a learned embedding table to look up the previous token and projects
    through a low-rank bottleneck to produce full-vocabulary logits.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        low_rank_dim: int,
    ) -> None:
        super().__init__()
        self.markov_emb = nn.Embedding(vocab_size, low_rank_dim)
        self.markov_proj = nn.Linear(low_rank_dim, vocab_size, bias=False)

    def forward(
        self,
        prev_token_ids: torch.Tensor,  # [num_blocks, block_size]
    ) -> torch.Tensor:  # [num_blocks, block_size, vocab_size]
        """Produce bigram logits from previous-token IDs."""
        emb = self.markov_emb(prev_token_ids)  # [num_blocks, block_size, rank]
        return self.markov_proj(emb)  # [num_blocks, block_size, vocab_size]
