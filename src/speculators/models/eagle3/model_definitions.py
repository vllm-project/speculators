# ruff: noqa: ERA001
import copy
import math
from typing import Any, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from transformers import Cache, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from speculators.train.distributed import get_sp_ring_group, get_sp_ulysses_group

def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )


class LlamaDecoderEagle3FirstLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        # Run original init
        super().__init__(config, layer_idx)

        # Apply Eagle3 modifications
        self.norm_before_residual = norm_before_residual
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn.q_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.k_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.v_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore[valid-type]
    ) -> torch.Tensor:

        # ##### Start of Eagle3 modifications #####

        # hidden_states are cat([embeds, hidden], dim=-1)
        # so residual should be hidden part only, and embeds should be normalized
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        # Apply norms
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        hidden_states = torch.cat([embeds, hidden], dim=-1)
        if torch.__version__ >= "2.10":
            # As of `torch==2.10`, compile attempts to fuse together too many
            # ops, resulting in a fused kernel that exceeds shared memory limits
            # For now, we force a graph break to prevent this
            # https://github.com/pytorch/pytorch/issues/175250
            torch._dynamo.graph_break()  # noqa: SLF001

        # ##### End of Eagle3 modifications #####

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states  # noqa: RET504


def basic_extract_local(value, rank, world_size, *args, **kwargs):
    return value.chunk(world_size, dim=1)[rank].detach().clone()


class NormLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        orig_max_position=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        # Llama3 style rotary embedding frequency scaling
        if all(
            v is not None
            for v in [
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            ]
        ):
            self.scaling_factor = scaling_factor
            self.low_freq_factor = low_freq_factor
            self.high_freq_factor = high_freq_factor
            self.orig_max_position = orig_max_position

            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq

            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
            else:
                smooth = 0

            new_freqs = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / self.scaling_factor,
                    (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
                ),
            )
            inv_freq = new_freqs

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings + 20,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaUSPFlashAttention(nn.Module):
    """
    LlamaUSPFlashAttention with Trainable Ring Attention & Correct Eagle3 Branch Merging.
    """

    def __init__(self, config):
        super().__init__()
        assert (
            dist.is_initialized()
        ), f"LlamaUSPAttention requires torch.distributed; call init_distributed first."
        self.rank = torch.distributed.get_rank()
        self.ring_pg = get_sp_ring_group()
        self.ulysses_pg = get_sp_ulysses_group()
        self.sp_ring_degree = torch.distributed.get_world_size(self.ring_pg)
        self.sp_ulysses_degree = torch.distributed.get_world_size(self.ulysses_pg)
        self.world_size = torch.distributed.get_world_size()
        self.ring_group_ranks = dist.get_process_group_ranks(self.ring_pg)
        self.ring_rank = torch.distributed.get_rank(self.ring_pg)

        self.attention_impl = self.ring_attention_hybrid_masked
        self.extract_func = basic_extract_local
        self.scatter_idx = 2
        self.gather_idx = 1
        self.use_sync = False
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(
            self.hidden_size * 2, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = NormLlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=getattr(self.config, "rope_theta", 10000),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()  # bs, seq_len, hidden_size
        local_q_len = q_len
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            query_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        ).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        key_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            key_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        ).transpose(1, 2)

        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            value_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        ).transpose(1, 2)
        q_len = q_len * self.sp_ring_degree * self.sp_ulysses_degree

        if self.sp_ring_degree > 1:
            
            # Standard RoPE: [bs, seq_len] -> split dim 1
            position_ids = position_ids.chunk(self.sp_ring_degree, dim=1)[
                self.ring_rank
            ].clone()

        # bs, shard_seqlen, hc, hs

        cos, sin = self.rotary_emb(query_states, seq_len=q_len)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        cache_k = [key_states]
        cache_v = [value_states]
        attn_output = self.attention_impl(
            query_states, attention_mask, cache_k, cache_v, q_len
        )

        attn_output = SeqAllToAll4D.apply(
            self.ulysses_pg,
            attn_output,
            self.gather_idx,
            self.scatter_idx,
            self.use_sync,
        )

        attn_output = attn_output.reshape(
            bsz, local_q_len, self.head_dim * self.num_heads
        )
        attn_output = self.o_proj(attn_output)
        return attn_output

    def ring_attention_hybrid_masked(
        self,
        local_q: torch.Tensor,  # [1, H, Local_S, D]
        attention_mask: torch.Tensor,  # [1, 1, Global_S, Global_S] (global Mask)
        cache_k: List[torch.Tensor],
        cache_v: List[torch.Tensor],
        q_len: int,  # Global_S
    ):
        group = self.ring_pg
        rank = self.ring_rank
        world_size = self.sp_ring_degree

        # bs=1
        _, H, local_seq_len, head_dim = local_q.shape
        scale = 1.0 / math.sqrt(head_dim)

        # global Q index
        start_q_idx = rank * local_seq_len
        end_q_idx = start_q_idx + local_seq_len

        # =============================================================
        # [FP32] 1. init Online Softmax static to float32
        # =============================================================
        m = torch.full(
            (1, H, local_seq_len),
            float("-inf"),
            device=local_q.device,
            dtype=torch.float32,
        )
        l = torch.zeros(
            (1, H, local_seq_len), device=local_q.device, dtype=torch.float32
        )
        acc = torch.zeros(
            (1, H, local_seq_len, head_dim), device=local_q.device, dtype=torch.float32
        )

        # =============================================================
        # Phase 1: Extras (cache_k[1:]) - local compute
        # =============================================================
        num_extras = len(cache_k) - 1
        if num_extras > 0:
            for i in range(1, len(cache_k)):
                ki = repeat_kv(cache_k[i], self.num_key_value_groups)  # bf16
                vi = repeat_kv(cache_v[i], self.num_key_value_groups)  # bf16

                score_i = (local_q * ki).sum(dim=-1).to(torch.float32) * scale

                # [FP32] Online Softmax Update
                m_new = torch.maximum(m, score_i)
                alpha = torch.exp(m - m_new)
                p_i = torch.exp(score_i - m_new)  # fp32

                l_new = l * alpha + p_i

                # [FP32] Accumulation
                # acc(fp32) = acc * alpha + p_i * vi(bf16->fp32)
                acc = acc * alpha.unsqueeze(-1)
                acc += p_i.unsqueeze(-1) * vi.to(torch.float32)

                m, l = m_new, l_new

        # =============================================================
        # Phase 2: Main Sequence (cache_k[0]) - Ring Attention
        # =============================================================
        local_k0 = repeat_kv(cache_k[0], self.num_key_value_groups)
        local_v0 = repeat_kv(cache_v[0], self.num_key_value_groups)

        curr_k, curr_v = local_k0, local_v0
        next_k, next_v = torch.empty_like(local_k0), torch.empty_like(local_v0)

        for step in range(world_size):
            # 2.1 communicate
            if step < world_size - 1:
                # ring group rank to global rank for isend and irecv
                send_dst = self.ring_group_ranks[(rank + 1) % world_size]
                recv_src = self.ring_group_ranks[(rank - 1 + world_size) % world_size]
                reqs = []

                if rank % 2 == 0:
                    reqs.append(dist.isend(curr_k, dst=send_dst, group=group))
                    reqs.append(dist.isend(curr_v, dst=send_dst, group=group))
                    reqs.append(dist.irecv(next_k, src=recv_src, group=group))
                    reqs.append(dist.irecv(next_v, src=recv_src, group=group))
                else:
                    reqs.append(dist.irecv(next_k, src=recv_src, group=group))
                    reqs.append(dist.irecv(next_v, src=recv_src, group=group))
                    reqs.append(dist.isend(curr_k, dst=send_dst, group=group))
                    reqs.append(dist.isend(curr_v, dst=send_dst, group=group))

            # 2.2 get index
            block_rank = (rank - step + world_size) % world_size
            start_k_idx = block_rank * local_seq_len
            end_k_idx = start_k_idx + local_seq_len

            # 2.3 [FP32] MatMul Score
            attn_block = torch.matmul(local_q, curr_k.transpose(2, 3)) * scale

            # 2.4 [FP32] Mask slice
            mask_slice = attention_mask[
                :, :, start_q_idx:end_q_idx, start_k_idx:end_k_idx
            ]
            if mask_slice.dtype != torch.float32:
                mask_slice = mask_slice.to(torch.float32)
            if mask_slice.device != attn_block.device:
                mask_slice = mask_slice.to(attn_block.device)

            attn_block = attn_block + mask_slice

            # 2.5 [FP32] Online Softmax Update
            m_block = attn_block.max(dim=-1).values  # fp32
            m_new = torch.maximum(m, m_block)

            alpha = torch.exp(m - m_new)
            p_block = torch.exp(attn_block - m_new.unsqueeze(-1))  # fp32

            l_new = l * alpha + p_block.sum(dim=-1)

            acc = acc * alpha.unsqueeze(-1)
            acc += torch.matmul(p_block, curr_v.to(torch.float32))

            m, l = m_new, l_new

            if step < world_size - 1:
                for req in reqs:
                    req.wait()
                curr_k, curr_v = next_k.clone(), next_v.clone()

        # =============================================================
        # 3. Finalize
        # =============================================================
        final_output = acc / (l.unsqueeze(-1) + 1e-6)

        return final_output.to(local_q.dtype).transpose(1, 2).contiguous()


class NormLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
    ):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaUSPFlashAttention(config=config)
        # self.mlp = LlamaMLP(config)
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
        """

        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        # Apply norms
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        hidden_states = torch.cat([embeds, hidden], dim=-1)
        
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states  # noqa: RET504


class ModelComponents(NamedTuple):
    first_layer_class: type
    decoder_layer_class: type
    norm_decoder_layer_class: type
    norm_class: type
    rotary_emb_class: type


model_classes: dict[str, ModelComponents] = {
    "llama": ModelComponents(
        LlamaDecoderEagle3FirstLayer,
        LlamaDecoderLayer,
        NormLlamaDecoderLayer,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
    ),
}
