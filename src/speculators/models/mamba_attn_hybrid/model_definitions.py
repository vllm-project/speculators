import torch
from torch import nn
from torch.nn.functional import silu, softplus
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from speculators.models.attention import flex_attention_forward
from speculators.models.dflash.model_definitions import apply_rotary_pos_emb

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

__all__ = [
    "AttentionSubLayer",
    "MLPSubLayer",
    "MambaSubLayer",
]


def ssd_scan(x, dt, a, b, c, d, initial_states):
    heads_per_group = x.shape[2] // b.shape[2]
    b = b.repeat_interleave(heads_per_group, dim=2)
    c = c.repeat_interleave(heads_per_group, dim=2)
    state = initial_states
    outputs = []
    for t in range(x.shape[1]):
        decay = torch.exp(dt[:, t] * a)[..., None, None]
        update = dt[:, t][..., None, None] * (x[:, t][..., None] * b[:, t][:, :, None])
        state = decay * state + update
        outputs.append((state * c[:, t][:, :, None]).sum(-1) + d[:, None] * x[:, t])
    return torch.stack(outputs, dim=1)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float() * silu(gate.float())
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class MambaMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        d_state: int,
        n_groups: int,
        conv_kernel: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.n_groups = n_groups
        self.inner_size = num_heads * head_dim
        self.conv_dim = self.inner_size + 2 * n_groups * d_state

        self.in_proj = nn.Linear(
            hidden_size, self.inner_size + self.conv_dim + num_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.norm = GatedRMSNorm(self.inner_size)
        self.out_proj = nn.Linear(self.inner_size, hidden_size, bias=False)

    @torch.compiler.disable
    def forward(
        self, hidden_states: torch.Tensor, initial_states: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        gate, xbc, dt = torch.split(
            self.in_proj(hidden_states),
            [self.inner_size, self.conv_dim, self.num_heads],
            dim=-1,
        )
        xbc = silu(self.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = torch.split(
            xbc,
            [
                self.inner_size,
                self.n_groups * self.d_state,
                self.n_groups * self.d_state,
            ],
            dim=-1,
        )
        x = x.reshape(batch, length, self.num_heads, self.head_dim)
        b = b.reshape(batch, length, self.n_groups, self.d_state)
        c = c.reshape(batch, length, self.n_groups, self.d_state)
        a = -torch.exp(self.A_log.float())

        if mamba_chunk_scan_combined is not None and x.is_cuda:
            y = mamba_chunk_scan_combined(
                x,
                dt,
                a,
                b,
                c,
                chunk_size=length,
                D=self.D,
                dt_bias=self.dt_bias,
                initial_states=initial_states,
                dt_softplus=True,
            )
        else:
            dt = softplus(dt.float() + self.dt_bias.float())
            y = ssd_scan(
                x.float(),
                dt,
                a,
                b.float(),
                c.float(),
                self.D.float(),
                initial_states.float(),
            ).to(hidden_states.dtype)

        y = self.norm(y.reshape(batch, length, self.inner_size), gate)
        return self.out_proj(y)


class VerifierKVAttention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        q_size = config.num_attention_heads * self.head_dim
        kv_size = config.num_key_value_heads * self.head_dim
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=bias)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        verifier_kv: tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        shape = (bsz, q_len, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = torch.cat([verifier_kv[0], k], dim=2)
        v = torch.cat([verifier_kv[1], v], dim=2)
        attn_output, _ = flex_attention_forward(
            self, q, k, v, attention_mask, scaling=self.scaling
        )
        return self.o_proj(attn_output.reshape(bsz, q_len, -1))


class MambaSubLayer(nn.Module):
    def __init__(self, config, mamba_config) -> None:
        super().__init__()
        self.ssm = MambaMixer(
            config.hidden_size,
            num_heads=mamba_config.mamba_num_heads,
            head_dim=mamba_config.mamba_head_dim,
            d_state=mamba_config.mamba_d_state,
            n_groups=mamba_config.mamba_n_groups,
            conv_kernel=mamba_config.mamba_conv_kernel,
        )
        self.ssm_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, initial_states: torch.Tensor
    ) -> torch.Tensor:
        blocks = hidden_states.view(
            initial_states.shape[0], -1, hidden_states.shape[-1]
        )
        blocks = blocks + self.ssm(self.ssm_norm(blocks), initial_states)
        return blocks.view_as(hidden_states)


class AttentionSubLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = VerifierKVAttention(config, layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        verifier_kv: tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
    ) -> torch.Tensor:
        return hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            verifier_kv,
            position_embeddings,
            attention_mask,
        )


class MLPSubLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))
