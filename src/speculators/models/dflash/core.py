"""DFlash model structure and layer definitions.

This module defines the DFlash model architecture without algorithm implementation.
"""

from typing import ClassVar

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3MLP,
    GradientCheckpointingLayer,
)

from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.utils.loading import load_model_layers


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings (local implementation)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention for DFlash model."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types") and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    """DFlash decoder layer."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(SpeculatorModel):
    """DFlash draft model structure.

    This is a model structure definition without algorithm implementation.
    The forward method should be implemented by algorithm-specific code.
    """

    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["embed_tokens.weight"]  # type: ignore[misc]

    def __init__(self, config: DFlashSpeculatorConfig, t2d: torch.Tensor, d2t: torch.Tensor) -> None:
        super().__init__(
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.config = config
        self.register_buffer("t2d", t2d)  # shape: [verifier_vocab_size], bool
        self.register_buffer("d2t", d2t)  # shape: [draft_vocab_size], int offsets
        self.draft_vocab_size = config.draft_vocab_size

        # Load verifier embeddings and tokenizer
        self._setup_embeddings_and_mask_token(config.speculators_config.verifier, t2d)

        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Setup target layer IDs for auxiliary hidden states
        num_verifier_layers = config.transformer_layer_config.num_hidden_layers
        self.target_layer_ids = self._build_target_layer_ids(num_verifier_layers, config.num_hidden_layers)

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,
        )
        self.block_size = config.block_size

    def _build_target_layer_ids(self, num_target_layers: int, num_draft_layers: int) -> list[int]:
        """Build target layer IDs for auxiliary hidden states."""
        if num_draft_layers == 1:
            return [num_target_layers // 2]
        start = 1
        end = num_target_layers - 3
        span = end - start
        target_layer_ids = [
            int(round(start + (i * span) / (num_draft_layers - 1)))
            for i in range(num_draft_layers)
        ]
        return target_layer_ids

    def _setup_embeddings_and_mask_token(self, verifier_config, t2d):
        """Setup embeddings and mask_token_id from verifier."""
        from transformers import AutoTokenizer

        if verifier_config.name_or_path is None:
            raise ValueError("VerifierConfig `name_or_path` value is required.")

        # Load embedding weights
        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight"],
            verifier_config.name_or_path,
        )

        # Create embedding layer
        self.embed_tokens = nn.Embedding(
            self.config.transformer_layer_config.vocab_size,
            self.config.transformer_layer_config.hidden_size,
            padding_idx=getattr(self.config.transformer_layer_config, "pad_token_id", None),
        )
        default_dtype = self.embed_tokens.weight.dtype
        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        self.embed_tokens.load_state_dict({"weight": embed_tokens_weight.to(dtype=default_dtype)})
        self.embed_tokens.weight.requires_grad = False

        vocab_size = int(t2d.sum().item())
        lm_head_weight = verifier_weights["lm_head.weight"]

        self.lm_head = torch.nn.Linear(
            self.config.transformer_layer_config.hidden_size, vocab_size, bias=False
        )
        self.verifier_lm_head = torch.nn.Linear(
            self.config.transformer_layer_config.hidden_size, self.draft_vocab_size, bias=False
        )
        masked_lm_head_weight = lm_head_weight.to(
            device=t2d.device, dtype=default_dtype
        )[t2d.to(torch.bool), :]

        self.lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = masked_lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False

        # Load tokenizer to get mask_token_id
        tokenizer = AutoTokenizer.from_pretrained(verifier_config.name_or_path)
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        self.mask_token_id = tokenizer.mask_token_id
