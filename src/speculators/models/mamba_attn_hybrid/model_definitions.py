import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from speculators.models.mamba_attn_hybrid.attention_layer import (
    Qwen3DFlashAttention,
)
from speculators.models.mamba_attn_hybrid.mamba import MambaDraftMixer


class HybridMambaSubLayer(nn.Module):
    def __init__(
        self,
        config,
        *,
        d_state: int = 16,
        num_heads: int = 64,
        head_dim: int = 64,
        n_groups: int = 1,
        conv_kernel: int = 4,
        expand: int = 1,
        num_anchors: int = 0,
    ) -> None:
        super().__init__()
        self.ssm = MambaDraftMixer(
            hidden_size=config.hidden_size,
            d_state=d_state,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            expand=expand,
        )
        self.ssm_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_anchors = num_anchors

    @torch.compiler.disable
    def _run_ssm(
        self, h: torch.Tensor, ssm_state0: torch.Tensor | None
    ) -> torch.Tensor:
        return self.ssm(self.ssm_norm(h), initial_states=ssm_state0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ssm_state0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, AK, d = hidden_states.shape
        A = self.num_anchors
        K = AK // A
        h = hidden_states.reshape(A, K, d)
        h = h + self._run_ssm(h, ssm_state0)
        return h.reshape(1, AK, d)


class HybridAttnSubLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_kv: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
        position_ids: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            context_kv=context_kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        return residual + hidden_states


class HybridMLPSubLayer(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.mlp_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))
