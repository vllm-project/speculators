"""
MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DFlash2 architecture adapted from:
https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
"""

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    Qwen3Config,
)
from typing_extensions import Unpack

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer

__all__ = [
    "CandidateSelector",
    "GroupedDynamicCausalConv",
    "Qwen3DFlash2DecoderLayer",
    "grouped_dynamic_conv",
]


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta_kernel: torch.Tensor,
    base_kernel: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Apply a token-conditioned grouped causal convolution within draft blocks."""
    hidden_size = hidden_states.shape[-1]
    if hidden_size % group_size:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by group_size "
            f"({group_size})."
        )

    kernel_size = base_kernel.shape[0]
    if kernel_size > block_size:
        raise ValueError(
            f"kernel_size ({kernel_size}) cannot exceed block_size ({block_size})."
        )
    num_groups = hidden_size // group_size
    if base_kernel.shape != (kernel_size, hidden_size):
        raise ValueError(
            "base_kernel must have shape [kernel_size, hidden_size], got "
            f"{tuple(base_kernel.shape)}."
        )
    expected_delta_shape = (*hidden_states.shape[:-1], kernel_size, num_groups)
    if delta_kernel.shape != expected_delta_shape:
        raise ValueError(
            f"delta_kernel must have shape {expected_delta_shape}, got "
            f"{tuple(delta_kernel.shape)}."
        )

    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, num_groups, group_size)
    flat_delta = delta_kernel.reshape(-1, kernel_size, num_groups)
    positions = torch.arange(
        flat_hidden.shape[0], device=hidden_states.device
    ).remainder(block_size)
    output = torch.zeros_like(flat_hidden)

    for tap in range(kernel_size):
        if tap == 0:
            shifted = flat_hidden
        else:
            padding = flat_hidden.new_zeros(tap, num_groups, group_size)
            shifted = torch.cat([padding, flat_hidden[:-tap]], dim=0)
        coefficient = (
            base_kernel[tap].to(flat_hidden.dtype).view(1, num_groups, group_size)
            + flat_delta[:, tap, :, None]
        )
        valid = positions.ge(tap).view(-1, 1, 1)
        output = output + shifted * coefficient * valid

    return output.reshape(original_shape)


class GroupedDynamicCausalConv(nn.Module):
    """Input/output local convolutions driven by one hidden-state projection."""

    def __init__(
        self,
        hidden_size: int,
        *,
        block_size: int,
        kernel_size: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size "
                f"({group_size})."
            )
        if kernel_size > block_size:
            raise ValueError(
                f"kernel_size ({kernel_size}) cannot exceed block_size ({block_size})."
            )
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * kernel_size * self.num_groups,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize both convolutions as identity transforms."""
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve sublayer inputs and return its output-side dynamic kernel."""
        kernels = self.kernel_projection(hidden_states).view(
            *hidden_states.shape[:-1],
            2,
            self.kernel_size,
            self.num_groups,
        )
        prepared = grouped_dynamic_conv(
            hidden_states,
            kernels[..., 0, :, :],
            self.base_kernel[0],
            block_size=self.block_size,
            group_size=self.group_size,
        )
        return prepared, kernels[..., 1, :, :]

    def finish(
        self, hidden_states: torch.Tensor, delta_kernel: torch.Tensor
    ) -> torch.Tensor:
        """Convolve sublayer outputs with the saved output-side kernel."""
        return grouped_dynamic_conv(
            hidden_states,
            delta_kernel,
            self.base_kernel[1],
            block_size=self.block_size,
            group_size=self.group_size,
        )


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """DFlash decoder layer with local convolution around attention and MLP."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        block_size: int,
        conv_kernel_size: int,
        conv_group_size: int,
    ) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        conv_kwargs = {
            "block_size": block_size,
            "kernel_size": conv_kernel_size,
            "group_size": conv_group_size,
        }
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, **conv_kwargs
        )
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, **conv_kwargs)

    def reset_convolutions(self) -> None:
        """Restore the DFlash-equivalent identity initialization."""
        self.attention_conv.reset_parameters()
        self.mlp_conv.reset_parameters()

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, output_kernel = self.attention_conv.prepare(hidden_states)
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
        hidden_states = self.attention_conv.finish(hidden_states, output_kernel)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, output_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, output_kernel)
        return residual + hidden_states


class CandidateSelector(nn.Module):
    """Bilinear predecessor/hidden/candidate scorer used after unary top-k."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if top_k > vocab_size:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed vocab_size ({vocab_size})."
            )
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=initializer_range)
        nn.init.normal_(self.successor_codebook, std=initializer_range)
        nn.init.normal_(self.hidden_projection.weight, std=initializer_range)

    def context(
        self, hidden_states: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return the elementwise predecessor and hidden-state interaction."""
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        return predecessor * projected_hidden.to(predecessor.dtype)

    def transition_scores(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score selected candidate IDs."""
        context = self.context(hidden_states, predecessor_ids)
        successors = self.successor_codebook[candidate_ids.long()]
        return (context.unsqueeze(-2) * successors).sum(dim=-1)

    def score_candidates(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add transition scores to a selected subset of unary logits."""
        unary_scores = unary_logits.gather(-1, candidate_ids)
        transition_scores = self.transition_scores(
            hidden_states, predecessor_ids, candidate_ids
        )
        return unary_scores + transition_scores.to(unary_scores.dtype)

    def select(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unary top-k IDs and their selector-corrected edge scores."""
        candidate_ids = unary_logits.topk(top_k or self.top_k, dim=-1).indices
        scores = self.score_candidates(
            unary_logits,
            hidden_states,
            predecessor_ids,
            candidate_ids,
        )
        return candidate_ids, scores
