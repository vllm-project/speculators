from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    apply_rotary_pos_emb_interleave,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)
from typing_extensions import Unpack

if TYPE_CHECKING:
    from collections.abc import Callable


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,  # noqa: ARG001
    unsqueeze_dim=1,
):
    """Apply rotary position embeddings (local implementation)."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Implements the custom attention which injects the target models
    # hidden states into the kv cache.
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads  # type: ignore[operator]
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

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
        # Instead of computing the k and v matricies from the hidden states,
        # the target_hidden is injected into the kv cache, (shape is context
        # length + block size)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # This is the main difference from the usual attention mechanism.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        # note the length becomes context length + block size
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
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
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    attention_class: type[nn.Module] = Qwen3DFlashAttention

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = self.attention_class(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        # The main difference between this method and the qwen 3 layer it is
        # built from is that it
        # passes the extra hidden states to the self attention from the verifier model.
        # Note that target_hidden is not modified here.
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # type: ignore[operator]


def _apply_mla_rope(
    interleave: bool,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the MLA rope channels for DFlash's asymmetric Q/K lengths.

    ``cos``/``sin`` span the full context+draft length; Q covers only the
    trailing draft block. The upstream helpers expect Q and K of equal length,
    so each side is rotated separately (cheap: ``qk_rope_head_dim`` wide).
    """
    if not interleave:
        return apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    q_len = q_rot.size(-2)
    q_rot, _ = apply_rotary_pos_emb_interleave(
        q_rot, q_rot, cos[..., -q_len:, :], sin[..., -q_len:, :]
    )
    _, k_rot = apply_rotary_pos_emb_interleave(k_rot, k_rot, cos, sin)
    return q_rot, k_rot


class DFlashMLAAttention(DeepseekV3Attention):
    """Upstream DeepSeek-style MLA with DFlash dual-source KV injection.

    Projections, latent norms, ``rope_interleave`` and the YaRN mscale softmax
    scaling all come from :class:`DeepseekV3Attention`. Only ``forward`` is
    DFlash-specific, mirroring :class:`Qwen3DFlashAttention`: Q is projected
    from the draft block, while the compressed KV latent is projected from
    ``cat(target_hidden, draft)`` so the block attends over the verifier context.
    """

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Draft blocks attend bidirectionally under the DFlash block mask.
        self.is_causal = False
        # kv_b_proj already expands the latent to num_heads K/V heads.
        self.num_key_value_groups = 1
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

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
        bsz, q_len = hidden_states.shape[:-1]
        kv_len = target_hidden.shape[1] + q_len

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # DFlash: the KV latent sees the verifier context followed by the block.
        kv_input = torch.cat([target_hidden, hidden_states], dim=1)
        k_pass, k_rot = torch.split(
            self.kv_a_proj_with_mqa(kv_input),
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        k_pass = (
            self.kv_b_proj(self.kv_a_layernorm(k_pass))
            .view(bsz, kv_len, -1, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_pass, v = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_rot = k_rot.view(bsz, 1, kv_len, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = _apply_mla_rope(
            self.config.rope_interleave, q_rot, k_rot, cos, sin
        )
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot), dim=-1)

        # Flash/flex kernels need matching Q/K/V head dims; GLM-5 has
        # v_head_dim == qk_head_dim, DeepSeek-style verifiers do not.
        if self.v_head_dim != self.qk_head_dim:
            v = nn.functional.pad(v, (0, self.qk_head_dim - self.v_head_dim))

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

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
        if self.v_head_dim != self.qk_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights


class DFlashMLADecoderLayer(Qwen3DFlashDecoderLayer):
    """Dense DFlash block (RMSNorm + gated MLP) with MLA attention."""

    attention_class = DFlashMLAAttention


class DFlashModelComponents(NamedTuple):
    """Per-``model_type`` building blocks for the DFlash backbone.

    Attributes:
        decoder_layer_class: Draft decoder layer class.
        rotary_head_dim_attr: Config attribute holding the RoPE width, or
            ``None`` to use ``config.head_dim`` (full-head GQA RoPE). MLA
            drafts rotate only ``qk_rope_head_dim``.
    """

    decoder_layer_class: type
    rotary_head_dim_attr: str | None


dflash_model_classes: dict[str, DFlashModelComponents] = {
    "qwen3": DFlashModelComponents(
        decoder_layer_class=Qwen3DFlashDecoderLayer,
        rotary_head_dim_attr=None,
    ),
    # GLM-5.x drafts: dense MLA. RoPE spans only qk_rope_head_dim.
    # Aux-hidden projection stays ``fc`` / ``hidden_norm`` (same checkpoint
    # keys as Qwen3); serving loaders remap those names if they need to.
    "glm5_dspark": DFlashModelComponents(
        decoder_layer_class=DFlashMLADecoderLayer,
        rotary_head_dim_attr="qk_rope_head_dim",
    ),
}
