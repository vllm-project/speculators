"""One mHC-wrapped DSV4 draft decoder block.

Residual flow keeps ``hc_mult`` streams; each sublayer (latent attention, then MoE) is
wrapped by a :class:`~.hyper.HyperConnection`:

    residual = streams
    post, comb, x = attn_hc(streams);  x = attn_norm(x);  x = attn(x, …)
    streams = place(x, residual, post, comb)
    residual = streams
    post, comb, x = ffn_hc(streams);   x = ffn_norm(x);   x = ffn(x)
    streams = place(x, residual, post, comb)

The attention is the block-gamma draft attention (:class:`~.attention.LatentAttention`),
which additionally consumes the target-hidden context ``main_x`` and the rope
frequencies for the block and context positions.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import LatentAttention
from .hyper import HyperConnection, place
from .moe import MoE
from .norm import RMSNorm


class MhcDecoderBlock(nn.Module):
    """Latent-attention + MoE block with two-site hyper-connections."""

    attn: LatentAttention
    ffn: MoE
    attn_norm: RMSNorm
    ffn_norm: RMSNorm
    attn_hc: HyperConnection
    ffn_hc: HyperConnection

    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn = LatentAttention(cfg)
        self.ffn = MoE(cfg)
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn_hc = HyperConnection(cfg)
        self.ffn_hc = HyperConnection(cfg)

    def forward(
        self,
        streams: torch.Tensor,
        context_x: torch.Tensor,
        block_freqs: torch.Tensor,
        context_freqs: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``streams [N, gamma, hc, dim]`` -> ``[N, gamma, hc, dim]``.

        ``context_x [N, W, dim]`` is the shared target-hidden context (``main_x``).
        """
        residual = streams
        post, comb, x = self.attn_hc(streams)
        x = self.attn_norm(x)
        x = self.attn(x, context_x, block_freqs, context_freqs, attn_bias)
        streams = place(x, residual, post, comb)

        residual = streams
        post, comb, x = self.ffn_hc(streams)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return place(x, residual, post, comb)
