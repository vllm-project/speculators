"""Building blocks of the DNeo parallel drafter (arXiv:2611.21925)."""

from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    Qwen3Config,
    eager_attention_forward,
)
from typing_extensions import Unpack

from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    _rotate_half,
)
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer

__all__ = [
    "DNeoCausalHead",
    "FeatureEncoder",
    "Qwen3DNeoAttention",
    "Qwen3DNeoDecoderLayer",
    "create_prefix_only_mask_mod",
    "dneo_target_layer_ids",
]


def dneo_target_layer_ids(num_layers: int, num_features: int) -> list[int]:
    """The DNeo layer-selection protocol: the first target layer, then
    ``num_features - 1`` layers spaced evenly over the second half of the target
    (the last one included), where the hidden states most useful for drafting are.

    Ids follow speculators' convention for ``aux_hidden_state_layer_ids``: id ``i``
    is the hidden state after ``i`` decoder layers (``num_layers`` is the last). So
    the first layer is id 1 and the second half, layers ``num_layers // 2`` to
    ``num_layers - 1`` counted from 0, is ids ``num_layers // 2 + 1`` to
    ``num_layers``.
    """
    if num_features < 2:  # noqa: PLR2004
        raise ValueError(f"num_features must be >= 2, got {num_features}")
    start, end = num_layers // 2 + 1, num_layers
    if end - start + 1 < num_features - 1:
        raise ValueError(
            f"a {num_layers}-layer target has too few layers in its second half "
            f"for {num_features} features"
        )
    later = torch.linspace(start, end, num_features - 1).round().long().tolist()
    return [1, *later]


class FeatureEncoder(nn.Module):
    """Fuse the selected target hidden states into one context feature without
    mixing hidden coordinates (no "residual mixing").

    ``mofa``: Mixture-of-Features Attention, one softmax weight per target layer
    (``layer_logits``). ``diagonal``: one weight per target layer and coordinate
    (``layer_weights``, ``[num_layers, hidden]``; the restricted encoder). Both
    are special cases of a dense encoder whose ``[hidden, num_layers * hidden]``
    matrix is diagonal within each layer block;
    ``to_dense`` returns that matrix, which runtimes serving DFlash's dense
    ``fc`` load unchanged.
    """

    def __init__(self, num_layers: int, hidden_size: int, mode: str = "mofa"):
        super().__init__()
        if mode not in ("mofa", "diagonal"):
            raise ValueError(f"unknown feature encoder {mode!r}")
        self.mode, self.num_layers, self.hidden_size = mode, num_layers, hidden_size
        if mode == "mofa":
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        else:
            self.layer_weights = nn.Parameter(
                torch.full((num_layers, hidden_size), 1.0 / num_layers)
            )

    def weights(self) -> torch.Tensor:
        """``[num_layers, hidden]`` weights (MoFA's softmax broadcast over hidden)."""
        if self.mode == "mofa":
            return self.layer_logits.softmax(-1)[:, None].expand(-1, self.hidden_size)
        return self.layer_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(
            *hidden_states.shape[:-1], self.num_layers, self.hidden_size
        )
        return (x * self.weights().to(x.dtype)).sum(-2)

    @torch.no_grad()
    def to_dense(self) -> torch.Tensor:
        """The equivalent dense ``[hidden, num_layers * hidden]`` encoder matrix."""
        weights = self.weights().detach()
        dense = weights.new_zeros(self.hidden_size, self.num_layers, self.hidden_size)
        dense.diagonal(dim1=0, dim2=2).copy_(weights)
        return dense.view(self.hidden_size, self.num_layers * self.hidden_size)

    @classmethod
    def from_dense(cls, dense: nn.Linear) -> "FeatureEncoder":
        """A diagonal encoder holding a dense encoder's within-layer diagonals."""
        hidden = dense.out_features
        num_layers = dense.in_features // hidden
        if num_layers * hidden != dense.in_features:
            raise ValueError("the dense encoder's input is not whole target layers")
        encoder = cls(num_layers, hidden, mode="diagonal")
        with torch.no_grad():
            diag = dense.weight.view(hidden, num_layers, hidden).diagonal(
                dim1=0, dim2=2
            )
            encoder.layer_weights.copy_(diag)
        return encoder


def create_prefix_only_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
):
    """Flex-attention mask mod for prefix-only attention.

    Query block ``j`` (``block_size`` draft positions for ``anchor_positions[j]``)
    attends only to context positions of its document before the anchor, within
    ``sliding_window`` if given; never to draft positions. KV holds only the
    context, so ``kv_len == total_seq_len``.

    A query with no such position (an anchor at its document's first token, or a
    padding anchor slot, which ``select_anchors`` puts at position 0) attends to
    the anchor's own context position instead: a row with no key at all would make
    eager and SDPA attention NaN. Those rows carry no loss.

    Returns:
        mask_mod, q_len, kv_len
    """
    anchor_positions = anchor_positions.to(
        device=document_ids.device, dtype=torch.long
    ).contiguous()
    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )
    q_len = anchor_positions.numel() * block_size
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    previous_doc = torch.cat([document_ids.new_full((1,), -2), document_ids[:-1]])
    has_context = (document_ids == previous_doc) & (document_ids != -1)

    def prefix_mod(_b, _h, q_idx, kv_idx):
        q_anchor = query_anchor_positions[q_idx]
        q_doc = document_ids[q_anchor]
        same_doc = (q_doc == document_ids[kv_idx]) & (q_doc != -1)
        visible = same_doc & (kv_idx < q_anchor)
        if sliding_window is not None:
            query_position = q_anchor + q_idx % block_size
            visible = visible & (kv_idx > query_position - sliding_window)
        no_context = ~has_context[q_anchor]
        return visible | (no_context & (kv_idx == q_anchor))

    return prefix_mod, q_len, total_seq_len


class Qwen3DNeoAttention(Qwen3DFlashAttention):
    """DFlash attention with prefix-only keys: draft positions query the context
    features alone, so no key or value is computed over the draft block."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None:
            raise NotImplementedError("DNeo prefix-only attention keeps no KV cache")
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_proj(target_hidden).view(bsz, ctx_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(target_hidden).view(bsz, ctx_len, -1, self.head_dim)
        v = v.transpose(1, 2)
        # position_embeddings cover [context | draft block]: the queries take the
        # draft positions, the keys the context positions.
        cos, sin = (t.unsqueeze(1) for t in position_embeddings)
        q = q * cos[..., -q_len:, :] + _rotate_half(q) * sin[..., -q_len:, :]
        k = k * cos[..., :ctx_len, :] + _rotate_half(k) * sin[..., :ctx_len, :]
        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = self.o_proj(attn_output.reshape(bsz, q_len, -1))
        return attn_output, attn_weights


class Qwen3DNeoDecoderLayer(Qwen3DFlash2DecoderLayer):
    """DFlash-2 decoder layer (2-tap convolutions around attention and MLP),
    optionally with prefix-only attention."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        block_size: int,
        conv_kernel_size: int,
        conv_group_size: int,
        prefix_only_attention: bool,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            block_size=block_size,
            conv_kernel_size=conv_kernel_size,
            conv_group_size=conv_group_size,
        )
        if prefix_only_attention:
            self.self_attn = Qwen3DNeoAttention(config=config, layer_idx=layer_idx)


class DNeoCausalHead(nn.Module):
    """The fused DNeo causal head: a low-rank logit correction for each draft
    position, conditioned on the previous token.

    Two latent terms of rank ``r`` are combined with position-dependent weights
    before one up-projection to the vocabulary:

    - DSpark's term, an embedding of the previous token;
    - DFlash-2's context-aware term, the previous token's code times a
      projection of the position's hidden state. Projected through the shared
      vocabulary embedding, it scores the whole vocabulary rather than a top-k.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        hidden_size: int,
        rank: int,
        block_size: int,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        # DSpark's names for its vocabulary factors: the optimizers route them to
        # AdamW, as embeddings and an output head.
        self.markov_w1 = nn.Embedding(verifier_vocab_size, rank)
        self.predecessor_codebook = nn.Embedding(verifier_vocab_size, rank)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.markov_w2 = nn.Linear(rank, draft_vocab_size, bias=False)
        self.markov_scale = nn.Parameter(torch.ones(block_size))
        self.context_scale = nn.Parameter(torch.ones(block_size))
        nn.init.normal_(self.markov_w1.weight, std=0.01)
        nn.init.normal_(self.predecessor_codebook.weight, std=initializer_range)
        nn.init.normal_(self.hidden_projection.weight, std=initializer_range)

    def latent(
        self, prev_token_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """``[N, block, r]`` fused latent code for previous tokens ``[N, block]`` and
        hidden states ``[N, block, hidden]``."""
        prev = self.markov_w1(prev_token_ids.long())
        projected = self.hidden_projection(hidden_states).to(prev.dtype)
        context = self.predecessor_codebook(prev_token_ids.long()) * projected
        block = prev_token_ids.shape[-1]
        a = self.markov_scale[:block].to(prev.dtype)[:, None]
        b = self.context_scale[:block].to(prev.dtype)[:, None]
        return a * prev + b * context

    def forward(
        self, prev_token_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """``[N, block, draft_vocab]`` logit bias."""
        latent = self.latent(prev_token_ids, hidden_states)
        return self.markov_w2(latent.to(self.markov_w2.weight.dtype))
