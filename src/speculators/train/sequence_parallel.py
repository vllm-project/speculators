"""Ulysses Sequence Parallelism utilities.

Implements the all-to-all communication pattern for distributing attention
computation across GPUs by sharding the sequence dimension. Non-attention
operations (FC, MLP, norms) operate on local sequence chunks without
communication.

Reference: DeepSpeed-Ulysses (https://arxiv.org/abs/2309.14509)
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

BatchType = dict[str, Any]

# Module-level state for SP process groups
_sp_group: ProcessGroup | None = None
_dp_group: ProcessGroup | None = None
_sp_size: int = 1
_sp_rank: int = 0
_dp_size: int = 1
_dp_rank: int = 0


def init_sp_process_groups(sp_size: int) -> tuple[ProcessGroup, ProcessGroup]:
    """Initialize sequence parallel and data parallel process groups.

    SP groups use contiguous ranks (e.g. sp_size=2, world_size=4: {0,1}, {2,3}).
    DP groups use strided ranks (e.g. sp_size=2, world_size=4: {0,2}, {1,3}).

    Returns:
        Tuple of (sp_group, dp_group) for the current rank.
    """
    global _sp_group, _dp_group, _sp_size, _sp_rank, _dp_size, _dp_rank

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    dp_size = world_size // sp_size

    sp_group = None
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        pg = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            sp_group = pg

    dp_group = None
    for i in range(sp_size):
        dp_ranks = list(range(i, world_size, sp_size))
        pg = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = pg

    assert sp_group is not None
    assert dp_group is not None

    _sp_group = sp_group
    _dp_group = dp_group
    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _dp_size = dp_size
    _dp_rank = rank // sp_size

    return sp_group, dp_group


def get_sp_group() -> ProcessGroup | None:
    return _sp_group


def get_dp_group() -> ProcessGroup | None:
    return _dp_group


def get_sp_size() -> int:
    return _sp_size


def get_sp_rank() -> int:
    return _sp_rank


def get_dp_size() -> int:
    return _dp_size


def get_dp_rank() -> int:
    return _dp_rank


# ---------------------------------------------------------------------------
# All-to-all for attention redistribution
# ---------------------------------------------------------------------------


class _AllToAllSP(torch.autograd.Function):
    """Differentiable all-to-all for sequence parallelism.

    Scatters along one dimension and gathers along another.
    The backward pass reverses the operation.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        sp_group: ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = dist.get_world_size(sp_group)
        if world_size == 1:
            return input

        input_chunks = input.chunk(world_size, dim=scatter_dim)
        output_chunks = [torch.empty_like(c) for c in input_chunks]
        dist.all_to_all(output_chunks, list(input_chunks), group=sp_group)
        return torch.cat(output_chunks, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _AllToAllSP.apply(
                grad_output, ctx.sp_group, ctx.gather_dim, ctx.scatter_dim
            ),
            None,
            None,
            None,
        )


def all_to_all_sp(
    input: torch.Tensor,
    sp_group: ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Perform a differentiable all-to-all operation for sequence parallelism."""
    return _AllToAllSP.apply(input, sp_group, scatter_dim, gather_dim)


# ---------------------------------------------------------------------------
# GQA head replication
# ---------------------------------------------------------------------------


def maybe_replicate_kv_heads(
    key: torch.Tensor,
    value: torch.Tensor,
    num_kv_heads: int,
    sp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate KV heads when sp_size > num_kv_heads for GQA models.

    Args:
        key: [batch, num_kv_heads, seq_len, head_dim]
        value: [batch, num_kv_heads, seq_len, head_dim]
        num_kv_heads: Number of key-value heads.
        sp_size: Sequence parallel world size.
    """
    if sp_size <= num_kv_heads:
        if num_kv_heads % sp_size != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"sp_size ({sp_size})"
            )
        return key, value

    if sp_size % num_kv_heads != 0:
        raise ValueError(
            f"sp_size ({sp_size}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) when sp_size > num_kv_heads"
        )

    replication_factor = sp_size // num_kv_heads
    key = key.repeat_interleave(replication_factor, dim=1)
    value = value.repeat_interleave(replication_factor, dim=1)
    return key, value


# ---------------------------------------------------------------------------
# Ulysses attention function
# ---------------------------------------------------------------------------


def ulysses_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Ulysses SP attention: wraps flex_attention with all-to-all communication.

    Input Q, K, V have shape [batch, num_heads, local_seq, head_dim] (all heads,
    local sequence chunk). The all-to-all redistributes to [batch, local_heads,
    global_seq, head_dim] for attention, then reverses.

    The attention_mask (BlockMask) must cover the global sequence length.
    """
    from torch.nn.attention.flex_attention import flex_attention

    sp_group = get_sp_group()
    sp_size = get_sp_size()

    if sp_group is None or sp_size <= 1:
        from speculators.models.attention import flex_attention_forward

        return flex_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **_kwargs
        )

    num_kv_heads = key.shape[1]
    key, value = maybe_replicate_kv_heads(key, value, num_kv_heads, sp_size)

    # Pre-attention: scatter heads (dim=1), gather seq (dim=2)
    query = all_to_all_sp(query, sp_group, scatter_dim=1, gather_dim=2)
    key = all_to_all_sp(key, sp_group, scatter_dim=1, gather_dim=2)
    value = all_to_all_sp(value, sp_group, scatter_dim=1, gather_dim=2)

    num_q_local = query.shape[1]
    num_kv_local = key.shape[1]

    attn_output = flex_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=num_q_local != num_kv_local,
        scale=scaling,
    )

    # Post-attention: scatter seq (dim=2), gather heads (dim=1)
    attn_output = all_to_all_sp(attn_output, sp_group, scatter_dim=2, gather_dim=1)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# ---------------------------------------------------------------------------
# Data scatter — SP rank 0 scatters collated batches to all SP ranks
# ---------------------------------------------------------------------------

_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.bool: 5,
    torch.uint8: 6,
    torch.float64: 7,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def sp_scatter_batch(
    batch: BatchType | None,
    sp_group: ProcessGroup,
    sp_rank: int,
    sp_size: int,
    device: torch.device,
) -> BatchType:
    """Scatter a collated batch from SP rank 0 to all SP ranks.

    All SP ranks must call this collectively. SP rank 0 passes the full
    collated batch; other ranks pass None. Metadata (keys, shapes, dtypes)
    is broadcast so receivers can allocate buffers. Sequence-dim tensors
    are scattered (dim 1); `lengths` is broadcast (all ranks need the full
    tensor for mask creation).

    Args:
        batch: Full collated batch on SP rank 0, None on other ranks.
        sp_group: The SP process group.
        sp_rank: This rank's position within the SP group.
        sp_size: Number of ranks in the SP group.
        device: Device to place output tensors on.

    Returns:
        Sharded batch with local sequence chunks and ``global_seq_len`` key.
    """
    src = dist.get_process_group_ranks(sp_group)[0]

    # 1. Broadcast number of tensor entries
    if sp_rank == 0:
        assert batch is not None
        tensor_items = [(k, v) for k, v in batch.items() if isinstance(v, torch.Tensor)]
        n = torch.tensor([len(tensor_items)], dtype=torch.long, device=device)
    else:
        tensor_items = []
        n = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(n, src=src, group=sp_group)
    num_entries = n.item()

    result: BatchType = {}

    # 2. For each tensor, broadcast metadata then scatter/broadcast data
    for i in range(num_entries):
        # --- key name ---
        if sp_rank == 0:
            key, tensor = tensor_items[i]
            key_enc = list(key.encode("utf-8"))
            klen = torch.tensor([len(key_enc)], dtype=torch.long, device=device)
        else:
            klen = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(klen, src=src, group=sp_group)

        if sp_rank == 0:
            kbytes = torch.tensor(key_enc, dtype=torch.uint8, device=device)
        else:
            kbytes = torch.empty(klen.item(), dtype=torch.uint8, device=device)
        dist.broadcast(kbytes, src=src, group=sp_group)
        key = bytes(kbytes.cpu().tolist()).decode("utf-8")

        # --- tensor metadata: [ndim, dtype_code, is_broadcast] + shape ---
        if sp_rank == 0:
            tensor = tensor.to(device)
            is_broadcast = key == "lengths" or tensor.dim() < 2
            meta = torch.tensor(
                [tensor.ndim, _DTYPE_TO_CODE[tensor.dtype], int(is_broadcast)],
                dtype=torch.long,
                device=device,
            )
            if is_broadcast:
                shape_list = list(tensor.shape)
            else:
                shape_list = list(tensor.shape)
                shape_list[1] = shape_list[1] // sp_size
        else:
            meta = torch.empty(3, dtype=torch.long, device=device)
        dist.broadcast(meta, src=src, group=sp_group)

        ndim = meta[0].item()
        dtype = _CODE_TO_DTYPE[meta[1].item()]
        is_broadcast = bool(meta[2].item())

        if sp_rank == 0:
            shape_t = torch.tensor(shape_list, dtype=torch.long, device=device)
        else:
            shape_t = torch.empty(ndim, dtype=torch.long, device=device)
        if ndim > 0:
            dist.broadcast(shape_t, src=src, group=sp_group)
        out_shape = [int(s) for s in shape_t.tolist()] if ndim > 0 else []

        # --- data transfer ---
        if is_broadcast:
            if sp_rank == 0:
                buf = tensor
            else:
                buf = torch.empty(out_shape, dtype=dtype, device=device)
            dist.broadcast(buf, src=src, group=sp_group)
            result[key] = buf
        else:
            recv = torch.empty(out_shape, dtype=dtype, device=device)
            if sp_rank == 0:
                chunks = [c.contiguous() for c in tensor.chunk(sp_size, dim=1)]
            else:
                chunks = None
            dist.scatter(recv, chunks, src=src, group=sp_group)
            result[key] = recv

    # global_seq_len metadata
    if "hidden_states" in result:
        result["global_seq_len"] = result["hidden_states"].shape[1] * sp_size

    return result


def sp_data_iterator(
    dataloader,
    sp_group: ProcessGroup,
    sp_rank: int,
    sp_size: int,
    device: torch.device,
) -> Generator[BatchType, None, None]:
    """Iterate a DataLoader with SP scatter.

    SP rank 0 drives the DataLoader; all ranks call ``sp_scatter_batch``
    collectively. Only SP rank 0 loads/generates data.

    Args:
        dataloader: DataLoader instance. Only iterated on SP rank 0;
            other ranks pass it only for ``len()``.
        sp_group: The SP process group.
        sp_rank: This rank's position within the SP group.
        sp_size: Number of ranks in the SP group.
        device: Device to place output tensors on.
    """
    # Synchronize batch count
    count = torch.tensor(
        [len(dataloader) if sp_rank == 0 else 0], dtype=torch.long, device=device
    )
    dist.broadcast(count, src=dist.get_process_group_ranks(sp_group)[0], group=sp_group)
    num_batches = count.item()

    if sp_rank == 0:
        for batch in dataloader:
            yield sp_scatter_batch(batch, sp_group, sp_rank, sp_size, device)
    else:
        for _ in range(num_batches):
            yield sp_scatter_batch(None, sp_group, sp_rank, sp_size, device)
