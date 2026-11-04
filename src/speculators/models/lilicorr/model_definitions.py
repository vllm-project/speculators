"""LiLiCorr candidate-lattice head (arXiv:2608.20530).

Ported from NVIDIA ModelOpt's ``modeling_lilicorr.py`` (Apache-2.0). Submodule
names follow vLLM's ``vllm/model_executor/models/lilicorr.py`` so a saved
checkpoint loads there directly.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

__all__ = ["LiLiCorrHead"]

# [log_p, p, log_p - max log_p, rank / (k - 1), is_top1]
_NUM_CANDIDATE_FEATURES = 5


class LiLiCorrMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.up_proj = nn.Linear(input_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class LiLiCorrAttention(nn.Module):
    """Unmasked attention over one block's lattice with an additive bias.

    Parameters use ``nn.MultiheadAttention``'s layout (``in_proj_weight`` /
    ``in_proj_bias`` / ``out_proj``), which is what vLLM loads.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}."
            )
        self.num_heads = num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        q, k, v = (
            F.linear(x, self.in_proj_weight, self.in_proj_bias)
            .view(bsz, seq_len, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out_proj(out.transpose(1, 2).reshape(bsz, seq_len, hidden))


class LiLiCorrLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.attn_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.attn = LiLiCorrAttention(hidden_size, num_heads)
        self.mlp_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.mlp = LiLiCorrMLP(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), bias)
        return x + self.mlp(self.mlp_norm(x))


class LiLiCorrHead(nn.Module):
    """Scores the ``slots x k`` candidate lattice of every block in one pass.

    Each candidate gets unit-norm ``out`` and ``in`` vectors; a transition from
    candidate ``i`` at slot ``s`` to candidate ``j`` at slot ``s+1`` scores
    ``logit_scale * cos(out[s, i], in[s+1, j])``. Slot 0 is scored against an
    ``anchor_out`` vector built from the target context row before the block.
    """

    def __init__(
        self,
        *,
        model_hidden_size: int,
        block_size: int,
        candidate_topk: int,
        hidden_size: int | None,
        factor_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        logit_scale: float,
        vector_eps: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        hidden_size = hidden_size or model_hidden_size
        self.block_size = block_size
        self.num_slots = block_size - 1
        self.candidate_topk = candidate_topk
        self.logit_scale = logit_scale
        self.vector_eps = vector_eps

        self.token_proj = (
            nn.Identity()
            if hidden_size == model_hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_norm = nn.LayerNorm(_NUM_CANDIDATE_FEATURES)
        self.feature_mlp = LiLiCorrMLP(
            _NUM_CANDIDATE_FEATURES, hidden_size, hidden_size
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_slots, 1, hidden_size)
        )
        self.rank_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, candidate_topk, hidden_size)
        )
        self.relative_slot_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * block_size - 1)
        )
        self.same_slot_bias = nn.Parameter(torch.zeros(num_heads))
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            LiLiCorrLayer(hidden_size, num_heads, mlp_ratio, rms_norm_eps)
            for _ in range(num_layers)
        )
        self.output_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.factor_input_proj = nn.Linear(3 * hidden_size, hidden_size)
        self.out_head = nn.Linear(hidden_size, factor_dim)
        self.in_head = nn.Linear(hidden_size, factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, factor_dim)
        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02) -> None:
        # Linears get N(0, std) with zero bias. The lattice QKV keeps its xavier
        # init, and the embeddings and attention biases start at zero.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                nn.init.zeros_(module.bias)

    def _attention_bias(self, device: torch.device) -> torch.Tensor:
        slot_ids = torch.arange(self.num_slots, device=device).repeat_interleave(
            self.candidate_topk
        )
        rel = slot_ids[:, None] - slot_ids[None, :]
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        bias = bias + (rel == 0) * self.same_slot_bias[:, None, None]
        return bias.unsqueeze(0)  # [1, heads, S*K, S*K], broadcast over blocks

    def forward(
        self,
        token_embeddings: torch.Tensor,  # [N, S, K, H_model]
        candidate_log_probs: torch.Tensor,  # [N, S, K], descending rank order
        pass_hidden: torch.Tensor,  # [N, S, H_model]
        anchor_hidden: torch.Tensor,  # [N, H_model]
        anchor_valid: torch.Tensor,  # [N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_start [N, K], log_pair [N, S-1, K, K])`` in fp32."""
        num_blocks, num_slots, topk = candidate_log_probs.shape

        log_probs = candidate_log_probs.float()
        rank = torch.arange(topk, device=log_probs.device) / max(topk - 1, 1)
        features = torch.stack(
            [
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.amax(-1, keepdim=True),
                rank.expand_as(log_probs),
                (rank == 0).float().expand_as(log_probs),
            ],
            dim=-1,
        )

        x = self.token_proj(token_embeddings) + self.pass_hidden_proj(
            pass_hidden
        ).unsqueeze(-2)
        x = x + self.feature_mlp(self.feature_norm(features.to(x.dtype)))
        x = x + self.slot_embedding[0] + self.rank_embedding[0]
        x = x.reshape(num_blocks, num_slots * topk, -1)

        bias = self._attention_bias(x.device)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.output_norm(x).view(num_blocks, num_slots, topk, -1)

        # A zeroed input still projects to context_proj's bias; zero the output too.
        anchor = self.context_proj(anchor_hidden) * anchor_valid[:, None].to(x.dtype)
        anchor = self.anchor_norm(anchor)
        a = anchor[:, None, None, :].expand_as(x)
        factors = F.silu(self.factor_input_proj(torch.cat([x, a, x * a], dim=-1)))

        out_vec = F.normalize(self.out_head(factors), dim=-1, eps=self.vector_eps)
        in_vec = F.normalize(self.in_head(factors), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(
            self.anchor_out_head(anchor), dim=-1, eps=self.vector_eps
        )

        start = (anchor_out[:, None, :] * in_vec[:, 0]).sum(-1)
        pair = out_vec[:, :-1] @ in_vec[:, 1:].transpose(-1, -2)
        return self.logit_scale * start.float(), self.logit_scale * pair.float()
