import torch
from torch import nn


class DominoHead(nn.Module):
    """GRU-based correction head that refines draft model logits.

    Processes previous token embeddings through a GRU, concatenates the
    resulting hidden states with the original hidden states, and projects
    them to produce logit deltas that are added to the suffix portion of
    the base logits.

    Parameters:
        hidden_size: Size of the transformer hidden states.
        gru_hidden_dim: Hidden dimension of the prefix GRU.
        emb_dim: Intermediate embedding dimension for the projection head.
        draft_vocab_size: Size of the draft model's vocabulary.
    """

    def __init__(
        self,
        hidden_size: int,
        gru_hidden_dim: int,
        emb_dim: int,
        draft_vocab_size: int,
    ) -> None:
        super().__init__()
        self.prefix_gru = nn.GRU(
            hidden_size, gru_hidden_dim, 1, batch_first=True, bias=False
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_dim, emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim, draft_vocab_size, bias=False),
        )

    def forward(
        self,
        hidden_states_4d: torch.Tensor,
        base_logits_4d: torch.Tensor,
        prev_token_ids: torch.Tensor,
        suffix_start: int,
        embed_tokens: nn.Embedding,
    ) -> torch.Tensor:
        """Apply GRU-based prefix refinement to logits.

        Parameters:
            hidden_states_4d: Tensor of shape ``(batch, num_anchors, block_size, hidden_size)``.
            base_logits_4d: Tensor of shape ``(batch, num_anchors, block_size, vocab_size)``.
            prev_token_ids: Tensor of shape ``(batch, num_anchors * block_size)`` with previous token IDs.
            suffix_start: Position index within a block where the suffix begins.
            embed_tokens: Embedding layer for token lookup.

        Returns:
            Refined logits tensor with the same shape as ``base_logits_4d``.
        """
        batch, num_anchors, block_size, hidden_size = hidden_states_4d.shape
        flat_batch = batch * num_anchors

        prev_embeds = embed_tokens(prev_token_ids)
        prev_embeds_2d = prev_embeds.view(flat_batch, block_size, hidden_size)

        gru_out, _ = self.prefix_gru(prev_embeds_2d)

        hidden_2d = hidden_states_4d.view(flat_batch, block_size, hidden_size)
        cat_features = torch.cat([hidden_2d, gru_out], dim=-1)
        logit_deltas = self.embed_proj(cat_features)
        logit_deltas = logit_deltas.view(batch, num_anchors, block_size, -1)

        refined_logits = base_logits_4d.clone()
        refined_logits[:, :, suffix_start:] = (
            base_logits_4d[:, :, suffix_start:] + logit_deltas[:, :, suffix_start:]
        )
        return refined_logits
