"""Multi-head latent attention with a per-head learnable sink (DSV4 draft).

Two layers here:

* :func:`sink_block_attention` — the numerical core: a dense, non-causal
  sink-softmax over ``[context + block]`` KV. Vanilla SDPA cannot
  express the sink (it is an extra term in the softmax denominator), so this
  eager einsum form is the sink-correct path.

* :class:`LatentAttention` — the MLA projection stack (low-rank query
  ``wq_a→q_norm→wq_b`` + per-head RMS, single shared-KV ``wkv→kv_norm``,
  grouped low-rank output ``wo_a→wo_b``) plus the learnable ``sink``. Its
  training forward assembles per-anchor-block queries/keys/values and calls
  :func:`sink_block_attention`; it does not use a KV cache (teacher-forced).

Weight-name note: the released checkpoint stores these as ``wq_a`` / ``wq_b`` /
``wkv`` / ``wo_a`` / ``wo_b`` / ``attn_sink``; the loader maps those onto the
descriptive attribute names here.
"""

from __future__ import annotations

import torch
from torch import nn

from .norm import RMSNorm, UnweightedRMSNorm
from .rotary import apply_rotary_emb

# k/v arrive as [N, Sk, D]: one shared KV head, no head axis.
_SHARED_KV_RANK = 3
# Cap on the fp32 logit block a single attention chunk may allocate. The dense
# [N, Sq, H, Sk] intermediates are the largest tensors in the model: at 512 anchors over
# a 22k-token context they are ~18 GB each, and three of them are live at once without
# chunking. Chunking the query axis is exact -- the softmax normalizes per query row --
# and leaves only the probabilities retained for backward, which activation
# checkpointing then removes as well.
_ATTN_CHUNK_BYTES = 1 << 30


def sink_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense non-causal sink-softmax attention (fp32 accumulation).

    Shapes: ``q [N, Sq, H, D]``, ``k/v [N, Sk, D]`` (a SINGLE shared KV head -- the DSV4
    draft's MLA has one KV head shared across the H query heads), ``sink [H]``, optional
    ``attn_bias [N, Sq, Sk]`` (or broadcastable) added to the logits *before* the sink
    term. Returns ``[N, Sq, H, D]``.

    Shared-KV einsums (``nkd``, not ``nkhd``) contract the single KV head against every
    query head directly -- identical to broadcasting K/V to H heads first, but without
    materializing the H-times-larger ``[N, Sk, H, D]`` tensor, and it is the shape a
    fused shared-KV kernel expects.

    The sink is a synthetic key whose logit is the per-head ``sink`` scalar; it
    contributes to the softmax denominator but nothing to the value sum:
    ``p_j = exp(s_j) / (Sum_j exp(s_j) + exp(sink))``.

    The query axis is processed in chunks sized by :data:`_ATTN_CHUNK_BYTES`. The
    softmax normalizes per query row, so chunking changes nothing but floating-point
    associativity.
    """
    # Loud guard against a stale caller that pre-expands K/V to H heads.
    if k.dim() != _SHARED_KV_RANK:
        raise ValueError(
            f"shared-KV sink attention wants k/v [N, Sk, D]; got {tuple(k.shape)}"
        )
    n, sq, h, _ = q.shape
    sk = k.shape[1]
    per_row = h * sk * 4  # one fp32 logit block for a single query position
    chunk = max(1, _ATTN_CHUNK_BYTES // max(per_row, 1))
    if chunk >= sq:
        return _attend(q, k, v, sink, scale, attn_bias)
    outs = [
        _attend(
            q[:, i : i + chunk],
            k,
            v,
            sink,
            scale,
            _slice_bias(attn_bias, i, i + chunk, sq),
        )
        for i in range(0, sq, chunk)
    ]
    return torch.cat(outs, dim=1)


def _slice_bias(
    attn_bias: torch.Tensor | None, start: int, stop: int, sq: int
) -> torch.Tensor | None:
    """Take the query slice of an additive bias, leaving a broadcast one alone."""
    if attn_bias is None or attn_bias.shape[-2] != sq:
        return attn_bias
    return attn_bias[..., start:stop, :]


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor | None,
) -> torch.Tensor:
    s = torch.einsum("nqhd,nkd->nqhk", q.float(), k.float()) * scale
    if attn_bias is not None:
        s = s + attn_bias.float().unsqueeze(2)  # [N, Sq, 1, Sk] broadcast over heads
    sink_h = sink.float().view(1, 1, -1, 1)
    row_max = torch.maximum(s.max(dim=-1, keepdim=True).values, sink_h)
    e = torch.exp(s - row_max)
    denom = e.sum(dim=-1, keepdim=True) + torch.exp(sink_h - row_max)
    p = e / denom
    return torch.einsum("nqhk,nkd->nqhd", p, v.float()).to(q.dtype)


class LatentAttention(nn.Module):
    """MLA + per-head sink for one draft layer (teacher-forced, no KV cache).

    ``forward`` takes the block hidden states (queries) and the context+block
    hidden states (keys/values source) already assembled by the caller, plus the
    precomputed rope frequencies for each, and returns the attention output projected
    back to ``hidden_size``.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.eps = cfg.rms_norm_eps
        self.scale = cfg.head_dim**-0.5

        self.wq_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.num_heads * cfg.head_dim, bias=False)
        self.q_head_norm = UnweightedRMSNorm(cfg.rms_norm_eps)  # per-head RMS on q
        self.wkv = nn.Linear(
            cfg.hidden_size, cfg.head_dim, bias=False
        )  # single shared KV head
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.wo_a = nn.Linear(
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
            cfg.o_groups * cfg.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size, bias=False
        )
        self.attn_sink = nn.Parameter(torch.zeros(cfg.num_heads))

    def project_q(self, block_x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """block_x [N, Sq, dim] -> q [N, Sq, H, head_dim], rope on the tail."""
        rd = self.rope_head_dim
        q = self.q_norm(self.wq_a(block_x))
        q = self.wq_b(q).unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_head_norm(q)
        rope = apply_rotary_emb(q[..., -rd:], freqs_cis)
        return torch.cat([q[..., :-rd], rope], dim=-1)

    def project_kv(self, kv_x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """kv_x [N, Sk, dim] -> kv [N, Sk, head_dim], one shared KV head."""
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(kv_x))
        rope = apply_rotary_emb(kv[..., -rd:], freqs_cis)
        return torch.cat([kv[..., :-rd], rope], dim=-1)

    def forward(
        self,
        block_x: torch.Tensor,
        context_x: torch.Tensor,
        block_freqs: torch.Tensor,
        context_freqs: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block-gamma draft attention (teacher-forced, no KV cache).

        ``block_x [N, gamma, dim]`` are the draft block hidden states (queries +
        block keys/values); ``context_x [N, W, dim]`` is the target-hidden context
        window (``main_x``) providing the sliding-window keys/values. Each block query
        attends densely (non-causal) to ``[context | block]`` with the per-head sink.
        Returns ``[N, gamma, dim]``.
        """
        q = self.project_q(block_x, block_freqs)  # [N, gamma, H, D]
        kv_ctx = self.project_kv(context_x, context_freqs)  # [N, W, D]
        kv_blk = self.project_kv(block_x, block_freqs)  # [N, gamma, D]
        kv = torch.cat(
            [kv_ctx, kv_blk], dim=1
        )  # [N, W+gamma, D] (single shared KV head)
        # Shared-KV: pass the single KV head directly (NO .expand to H heads). The
        # nkd einsums broadcast it over the query heads -> identical output,
        # without ever materializing the H×-larger [N, Sk, H, D] tensor (−~2.1 GB, ~20×
        # forward).
        o = sink_block_attention(q, kv, kv, self.attn_sink, self.scale, attn_bias)
        return self.combine_output(o, block_freqs)

    def combine_output(
        self, o: torch.Tensor, q_freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """o [N, Sq, H, head_dim] -> hidden [N, Sq, dim] via grouped low-rank
        projection.

        First de-rotate the output's rope slice (K=V shared rotation), then the grouped
        ``wo_a`` einsum + ``wo_b`` mix.
        """
        rd = self.rope_head_dim
        derot = apply_rotary_emb(o[..., -rd:], q_freqs_cis, inverse=True)
        o = torch.cat([o[..., :-rd], derot], dim=-1)
        n, sq = o.shape[0], o.shape[1]
        o = o.reshape(n, sq, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.cfg.o_lora_rank, -1)
        o = torch.einsum("nsgd,grd->nsgr", o, wo_a)
        return self.wo_b(o.flatten(2))
