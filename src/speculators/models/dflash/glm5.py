"""Dense MLA decoder layers for GLM-5.x DFlash / DSpark drafts.

Q is projected from the draft block hidden states; K/V are projected from
``cat(target_context, draft)`` through DeepSeek-style MLA (no DSA indexer).
RoPE is interleaved pair rotation ``(0,1), (2,3), ...`` (vLLM
``get_rope(..., is_neox_style=False)``, GLM ``rope_interleave=true``) and is
applied only to the ``qk_rope_head_dim`` side channels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    Qwen3RMSNorm,
    eager_attention_forward,
)
from typing_extensions import Unpack

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer

__all__ = [
    "GLM5_DSPARK_ARCHITECTURE",
    "GLM5_MLA_MODEL_TYPES",
    "Glm5Config",
    "Glm5DFlashDecoderLayer",
    "Glm5DFlashMLAAttention",
    "apply_interleaved_rotary_pos_emb",
    "is_glm5_mla_config",
    "mla_kwargs_from_verifier",
]

# Serve-side vLLM-Ascend registration name (K3 DSpark uses K3DSparkForCausalLM).
GLM5_DSPARK_ARCHITECTURE = "Glm5DSparkForCausalLM"
# Current export name plus the legacy training ``model_type``.
GLM5_MLA_MODEL_TYPES = frozenset({"glm5_dspark", "glm5"})


class Glm5Config(Qwen3Config):
    """Qwen3-shaped decoder config plus GLM-5 Dense MLA dimensions.

    MLP / RMSNorm / RoPE machinery stay Qwen3-compatible so DFlash can reuse
    those modules. Attention is swapped for :class:`Glm5DFlashMLAAttention`.

    ``model_type`` is ``glm5_dspark`` so vLLM-Ascend can dispatch the draft
    independently of the GLM-5.2 verifier (``glm_moe_dsa``). Defaults lock the
    serveable 576-d MLA page spec: ``kv_lora_rank=512`` + ``qk_rope_head_dim=64``.
    """

    model_type = "glm5_dspark"

    def __init__(
        self,
        q_lora_rank: int | None = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int | None = None,
        qk_head_dim: int | None = None,
        rope_interleave: bool = True,
        **kwargs,
    ):
        # Drop a stale nested ``model_type`` so loading a legacy ``glm5``
        # checkpoint still constructs this class as ``glm5_dspark``.
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.model_type = "glm5_dspark"
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = (
            qk_nope_head_dim + qk_rope_head_dim
            if qk_head_dim is None
            else qk_head_dim
        )
        # GLM-5.2: v_head_dim == qk_head_dim (256), not qk_nope (192).
        self.v_head_dim = self.qk_head_dim if v_head_dim is None else v_head_dim
        self.rope_interleave = rope_interleave


def _register_glm5_config(model_type: str) -> None:
    try:
        AutoConfig.register(model_type, Glm5Config, exist_ok=True)
    except (TypeError, ValueError):
        # transformers without exist_ok, or model_type != Glm5Config.model_type
        try:
            AutoConfig.register(model_type, Glm5Config)
        except ValueError:
            pass


_register_glm5_config("glm5_dspark")
# Do not AutoConfig.register("glm5"): transformers requires the registered
# name to equal Glm5Config.model_type (now glm5_dspark). Legacy "glm5"
# checkpoints are still accepted by is_glm5_mla_config and the DFlash
# transformer_layer_config validator.


def is_glm5_mla_config(config: Any) -> bool:
    return getattr(config, "model_type", None) in GLM5_MLA_MODEL_TYPES


def mla_kwargs_from_verifier(verifier_config: PretrainedConfig) -> dict[str, Any]:
    """Copy Dense MLA ranks/head dims from a GLM-5 / DeepSeek-style verifier."""
    required = (
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
    )
    missing = [
        name for name in required if getattr(verifier_config, name, None) is None
    ]
    if missing:
        raise ValueError(
            "--draft-arch glm5 requires verifier MLA fields "
            f"{missing}. Use a GLM-5 / DeepSeek-style verifier, "
            "or --draft-arch qwen3."
        )
    qk_nope = int(verifier_config.qk_nope_head_dim)
    qk_rope = int(verifier_config.qk_rope_head_dim)
    qk_head_dim = getattr(verifier_config, "qk_head_dim", None)
    if qk_head_dim is None:
        qk_head_dim = qk_nope + qk_rope
    else:
        qk_head_dim = int(qk_head_dim)
        # Some transformers versions (e.g. 5.5.x GlmMoe) clobber
        # ``qk_rope_head_dim`` with ``head_dim`` / ``qk_nope``. Recover from
        # the qk_head_dim invariant: qk_head = nope + rope.
        if qk_nope + qk_rope != qk_head_dim:
            qk_rope = qk_head_dim - qk_nope
    v_head_dim = getattr(verifier_config, "v_head_dim", None)
    if v_head_dim is None:
        v_head_dim = qk_head_dim
    return {
        "q_lora_rank": int(verifier_config.q_lora_rank),
        "kv_lora_rank": int(verifier_config.kv_lora_rank),
        "qk_nope_head_dim": qk_nope,
        "qk_rope_head_dim": qk_rope,
        "v_head_dim": int(v_head_dim),
        "qk_head_dim": int(qk_head_dim),
        "rope_interleave": bool(getattr(verifier_config, "rope_interleave", True)),
    }


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """GPT-J / vLLM ``is_neox_style=False``: rotate pairs ``(0,1), (2,3), ...``."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_interleaved_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved (GPT-J) RoPE using HF NeoX-duplicated cos/sin.

    :class:`~transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding`
    emits ``cat(freqs, freqs)`` (split-half duplication). vLLM
    ``get_rope(..., is_neox_style=False)`` rotates pairs with the unique
    ``d/2`` frequencies. Repeat-interleave those freqs to
    ``[f0, f0, f1, f1, ...]`` and apply pair rotation so training matches GLM
    ``rope_interleave=true``.

    Q is rotated on the trailing draft segment only; K is rotated over the
    full prefix+draft length (same slicing as the Qwen3 DFlash NeoX helper).
    """
    rotary_dim = q.shape[-1]
    # Unique frequencies occupy the first half of the NeoX-duplicated cache.
    cos = cos[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (
        _rotate_interleaved(q) * sin[..., -q_len:, :]
    )
    k_embed = (k * cos) + (_rotate_interleaved(k) * sin)
    return q_embed, k_embed


class Glm5DFlashMLAAttention(nn.Module):
    """Dense MLA with DFlash dual-source KV injection.

    * Q is projected from draft (noise) hidden states only.
    * K/V are projected from ``cat(target_hidden, draft_hidden)``.
    * No DSA indexer: attention is dense under the DFlash block mask.
    """

    def __init__(self, config: Glm5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        q_lora_rank = getattr(config, "q_lora_rank", None)
        self.q_lora_rank = None if not q_lora_rank else int(q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = int(
            getattr(config, "qk_head_dim", None)
            or (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.scaling = 1.0 / math.sqrt(self.qk_head_dim)
        # K/V already have ``num_heads`` after MLA up-projection + k_rope expand.
        self.num_key_value_groups = 1
        self.head_dim = self.qk_head_dim
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = Qwen3RMSNorm(
                self.q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=config.attention_bias,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = Qwen3RMSNorm(
            self.kv_lora_rank, eps=config.rms_norm_eps
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is not None:
            return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        return self.q_proj(hidden_states)

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
        ctx_len = target_hidden.shape[1]
        kv_len = ctx_len + q_len

        q = self._project_q(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv_input = torch.cat([target_hidden, hidden_states], dim=1)
        compressed_kv, k_rope = torch.split(
            self.kv_a_proj_with_mqa(kv_input),
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(
            bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_nope = k_nope.transpose(1, 2)
        value = value.transpose(1, 2)
        k_rope = k_rope.unsqueeze(1)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_interleaved_rotary_pos_emb(q_rope, k_rope, cos, sin)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)

        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        # Flash/SDPA kernels require Q/K/V to share the last dim. Pad V when
        # v_head_dim != qk_head_dim (DeepSeek-style); GLM-5.2 they are equal.
        value_for_attn = value
        if self.v_head_dim != self.qk_head_dim:
            value_for_attn = nn.functional.pad(
                value, (0, self.qk_head_dim - self.v_head_dim)
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_for_attn = past_key_values.update(
                key_states, value_for_attn, self.layer_idx, cache_kwargs
            )

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
            query_states,
            key_states,
            value_for_attn,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if self.v_head_dim != self.qk_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output), attn_weights


class Glm5DFlashDecoderLayer(Qwen3DFlashDecoderLayer):
    attention_class = Glm5DFlashMLAAttention
