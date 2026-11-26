"""Tensor contracts for anchored block drafts."""

from typing import NamedTuple

import torch
from torch.nn.attention.flex_attention import BlockMask


class BlockLayout(NamedTuple):
    """Alignment of flattened draft slots with the original packed sequence.

    ``block_positions`` indexes source tokens, not rotary position IDs. Padded
    blocks have zero ``loss_mask``. With
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
        # Repeat the anchor and shift right: [a, b, c, d] -> [a, a, b, c].
        return torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)


class BackboneInputs(NamedTuple):
    """Inputs in DFlashBackbone.forward argument order."""

    token_embeddings: torch.Tensor
    target_hidden: torch.Tensor
    position_ids: torch.Tensor
    sliding_window_attn_mask: torch.Tensor | BlockMask | None
    full_attn_mask: torch.Tensor | BlockMask | None
