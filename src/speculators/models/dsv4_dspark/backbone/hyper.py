"""Manifold-constrained hyper-connections (mHC) for the DSV4 DSpark backbone.

Instead of a single residual stream, each block keeps ``hc_mult`` copies. At a sublayer
site (attention / MoE) a :class:`HyperConnection`:

  * collapses the ``hc_mult`` streams to one input for the sublayer (``pre``
    weights), and
  * returns per-stream placement weights ``post`` and a Sinkhorn-projected
    doubly-stochastic stream-mixing matrix ``comb`` used by :func:`place` to
    fold the sublayer output back into the multi-stream residual.

:class:`HyperHead` is the lighter final collapse (``hc_mult`` → 1) before the
shared norm + lm_head.

The math follows the reference exactly: ``pre = σ(·)+ε``, ``post = 2·σ(·)`` (no ε),
``comb = softmax(·)+ε`` then Sinkhorn-Knopp (one column normalization, then ``iters-1``
row/column passes).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .norm import UnweightedRMSNorm

# Small-normal init for the projections created with torch.empty.
_INIT_STD = 0.02


def _hyper_connection(
    module: HyperConnection, streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference mHC forward. ``streams [B, S, hc, D]`` -> ``(post, comb, collapsed)``.

    ``post [B, S, hc]``, ``comb [B, S, hc, hc]`` (doubly-stochastic),
    ``collapsed [B, S, D]``.
    """
    hc = module.hc_mult
    eps = module.hc_eps
    flat = module.input_norm(streams.flatten(start_dim=2).float())
    mix = functional.linear(flat, module.fn.float())
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = module.base.float().split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = module.scale.float().unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    # Sinkhorn-Knopp toward doubly-stochastic: start with one column pass (the softmax
    # already made rows sum ~1), then iters-1 row/column passes.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    return post, comb, collapsed


def place(
    out: torch.Tensor,
    residual_streams: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Fold a sublayer output back into the multi-stream residual.

    ``out [B, S, D]``, ``residual_streams [B, S, hc, D]``, ``post [B, S, hc]``,
    ``comb [B, S, hc, hc]`` -> new streams ``[B, S, hc, D]``:
    ``new[j] = post[j]·out + Σ_m comb[m, j]·residual[m]`` (matches the reference
    ``sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)``).
    """
    placed = post.unsqueeze(-1) * out.unsqueeze(-2)
    mixed = torch.sum(comb.unsqueeze(-1) * residual_streams.unsqueeze(-2), dim=2)
    return (placed + mixed).type_as(out)


class HyperConnection(nn.Module):
    """One mHC site (attention or FFN)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.input_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * cfg.hidden_size))
        nn.init.normal_(self.fn, std=_INIT_STD)
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))  # index 0=pre, 1=post, 2=comb

    def forward(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _hyper_connection(self, streams)


class HyperHead(nn.Module):
    """Final mHC collapse: ``[B, S, hc, D]`` -> ``[B, S, D]`` (pre weights only)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.input_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        self.hc_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * cfg.hidden_size)
        )
        nn.init.normal_(self.hc_fn, std=_INIT_STD)
        self.hc_base = nn.Parameter(torch.zeros(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.ones(1))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(streams.flatten(start_dim=2).float())
        mixes = functional.linear(flat, self.hc_fn.float())
        pre = (
            torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
            + self.hc_eps
        )
        return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
