"""Tensor contracts shared by block-drafting backbones and their callers.

These containers do not own modules, parameters, targets, or training losses.
Token IDs refer to the verifier vocabulary unless explicitly named otherwise.
"""

from typing import NamedTuple

import torch
from torch.nn.attention.flex_attention import BlockMask

AttentionMask = torch.Tensor | BlockMask | None


class BlockLayout(NamedTuple):
    """Alignment of flattened draft slots with the original packed sequence.

    ``block_positions`` indexes source tokens, not rotary position IDs. Invalid
    padded blocks have safe source indices and zero ``loss_mask``. With
    ``sample_from_anchor``, slot k predicts source token p+k+1; otherwise it
    predicts p+k and slot zero is an untrained anchor.
    """

    anchor_positions: torch.Tensor  # [num_blocks]
    block_positions: torch.Tensor  # [num_blocks * block_size]
    loss_mask: torch.Tensor  # [1, num_blocks * block_size]
    block_size: int
    sample_from_anchor: bool

    def predecessor_ids(
        self,
        input_ids: torch.Tensor,  # [1, total_seq_len]
    ) -> torch.Tensor:  # [num_blocks, block_size]
        """Teacher-forced predecessors with the anchor repeated in slot zero.

        Objectives such as XPress that need the token before the anchor must
        explicitly supply that document-aware predecessor instead.
        """
        tokens = input_ids[0, self.block_positions].reshape(-1, self.block_size)
        if self.sample_from_anchor:
            return tokens
        return torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)


class BackboneInputs(NamedTuple):
    """Prepared inputs for the original sequence and synthetic draft blocks.

    ``total_seq_len`` includes source padding. ``num_blocks`` is the allocated
    anchor count, including padded blocks, so each draft tensor contains
    ``num_blocks * block_size`` positions. Rotary positions cover both sequences.
    """

    token_embeddings: torch.Tensor  # [1, num_blocks * block_size, hidden_size]
    target_hidden: torch.Tensor  # [1, total_seq_len, num_target_layers * hidden_size]
    position_ids: torch.Tensor  # [1, total_seq_len + num_blocks * block_size]
    full_attn_mask: AttentionMask
    sliding_window_attn_mask: AttentionMask


class DraftFeatures(NamedTuple):
    """Normalized backbone features and shared vocabulary projection."""

    hidden: torch.Tensor  # [1, num_blocks * block_size, hidden_size]
    logits: torch.Tensor  # [1, num_blocks * block_size, draft_vocab_size]
    layout: BlockLayout
