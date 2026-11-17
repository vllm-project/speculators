"""DFlash transformer body, independent of vocabulary and training objectives."""

from collections.abc import Callable

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from speculators.models.components import AttentionMask
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.utils import flatten_rope_parameters


class DFlashBackbone(nn.Module):
    def __init__(
        self,
        tl_config: PretrainedConfig,
        num_target_layers: int,
        decoder_layer_factory: Callable[..., nn.Module] = Qwen3DFlashDecoderLayer,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                decoder_layer_factory(tl_config, layer_idx)
                for layer_idx in range(tl_config.num_hidden_layers)
            ]
        )
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(
                getattr(tl_config, "layer_types", None) or []
            )
            if layer_type == "sliding_attention"
        ]

        self.norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        rotary_config = flatten_rope_parameters(tl_config)
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            num_target_layers * tl_config.hidden_size,
            tl_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,  # [1, num_blocks * block_size, hidden_size]
        # [1, total_seq_len, num_target_layers * hidden_size]
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,  # [1, total_seq_len + num_blocks * block_size]
        # Dense masks: [1, 1, num_blocks * block_size,
        #               total_seq_len + num_blocks * block_size].
        # BlockMask has the same logical query/key lengths.
        sliding_window_attn_mask: AttentionMask = None,
        full_attn_mask: AttentionMask = None,
        **kwargs,
    ) -> torch.Tensor:  # [1, num_blocks * block_size, hidden_size]
        """Return normalized draft-slot hidden states, without logits or targets."""

        fc_output = self.fc(target_hidden)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(target_hidden, position_ids)

        hidden = token_embeddings
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(
                hidden_states=hidden,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return self.norm(hidden)
        # shape: [1, num_blocks * block_size, hidden_size]
