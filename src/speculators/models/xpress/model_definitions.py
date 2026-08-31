import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

__all__ = [
    "XPressRefinerHead",
]


class XPressRefinerHead(nn.Module):
    """Lightweight causal refiner producing a per-position logit bias."""

    # register_buffer widens the attribute to Tensor | Module; the annotation
    # tells the type checker which one it is.
    causal_tril: torch.Tensor

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        hidden_size: int,
        block_size: int,
        rank: int = 256,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.rank = rank
        self.block_size = block_size
        r = rank
        self.token_embed = nn.Embedding(verifier_vocab_size, r)
        self.down_h = nn.Linear(hidden_size, r, bias=False)
        self.down_g = nn.Linear(hidden_size, r, bias=False)
        self.in_proj = nn.Linear(3 * r, r, bias=False)
        self.mix_l = nn.Parameter(torch.zeros(r, block_size, block_size))
        mlp_hidden = r * mlp_ratio
        self.mlp_gate = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, r, bias=False)
        self.readout = nn.Linear(r, draft_vocab_size, bias=False)
        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_tril", tril, persistent=False)

    def hidden_cache(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pass-invariant hidden features ``[down_h(h) ; down_g(g)]``.

        hidden_states: ``[N, block, hidden]``. Returns ``[N, block, 2r]``.
        Computed once per block and reused across refine passes.
        """
        g = hidden_states.mean(dim=1, keepdim=True).expand_as(hidden_states)
        return torch.cat([self.down_h(hidden_states), self.down_g(g)], dim=-1)

    def block_bias(
        self,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        hcache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hcache is None:
            if hidden_states is None:
                raise ValueError("block_bias needs hidden_states or hcache")
            hcache = self.hidden_cache(hidden_states.to(self.down_h.weight.dtype))
        lat = self.token_embed(prev_token_ids.long())
        x = self.in_proj(torch.cat([hcache, lat.to(hcache.dtype)], dim=-1))
        # c[n,k,r] = a[n,k,r] + sum_{j<=k} L[r,k,j] a[n,j,r]
        l_eff = self.mix_l * self.causal_tril.to(self.mix_l.dtype)
        x = x + torch.einsum("ckj,njc->nkc", l_eff, x)
        x = x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))
        return self.readout(x)
