import logging
import os
from typing import Any, Optional

import torch
import torch.distributed as dist


local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_distributed = "LOCAL_RANK" in os.environ

logger = logging.getLogger("speculators")


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP


def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def maybe_setup_distributed_sp_ulysses(sp_ulysses_size: int, sp_ring_size: int) -> tuple[int, int, int, bool]:
    """Sets up distributed training if the process was launched with `torchrun`.
    If not, returns single process training.

    Based on of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    Args:
        sp_ulysses_size: Size of SP Ulysses group.
        sp_ring_size: Size of SP Ring group.

    Returns:
        tuple[int, int, int, bool]: Local rank, world size, rank, and is_distributed.
    """
    if not is_distributed:
        # No distributed training
        return 0, 1, 0, False

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert (
        world_size % (sp_ulysses_size * sp_ring_size) == 0
    ), f"World size ({world_size}) cannot be evenly divided by total SP size ({sp_ulysses_size*sp_ring_size})"

    draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "npu",
        (draft_dp_size, sp_ulysses_size * sp_ring_size),
        mesh_dim_names=("draft_dp", "sp"),
    )
    sp_ring_degree = sp_ring_size
    sp_ulysses_degree = sp_ulysses_size
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert (
        world_size % sp_degree == 0
    ), f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    for dp_rank in range(dp_degree):
        offset = dp_rank * sp_degree
        for i in range(num_ulysses_pgs):
            ulysses_ranks = list(
                range(
                    i * sp_ulysses_degree + offset,
                    (i + 1) * sp_ulysses_degree + offset,
                )
            )
            group = torch.distributed.new_group(ulysses_ranks)
            if rank in ulysses_ranks:
                ulyssess_pg = group

        for i in range(num_ring_pgs):
            ring_ranks = list(range(i + offset, sp_degree + offset, num_ring_pgs))
            group = torch.distributed.new_group(ring_ranks)
            if rank in ring_ranks:
                ring_pg = group
    sp_ulysses_group = ulyssess_pg
    sp_ring_group = ring_pg

    global _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _DRAFT_SP_GROUP = draft_device_mesh.get_group("sp")

    logger.info(
        f"Started distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )
    return local_rank, world_size, rank, True

def maybe_setup_distributed(enable_sp_ulysses: bool = False, sp_ulysses_size: int = 1, sp_ring_size: int = 1) -> tuple[int, int, int, bool]:
    """Sets up distributed training if the process was launched with `torchrun`.
    If not, returns single process training.

    Based on of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

    Args:
        enable_sp_ulysses: Whether to enable SP Ulysses sequence parallelism.
        sp_ulysses_size: Size of SP Ulysses group.
        sp_ring_size: Size of SP Ring group.

    Returns:
        tuple[int, int, int, bool]: Local rank, world size, rank, and is_distributed.
    """
    if not is_distributed:
        # No distributed training
        return 0, 1, 0, False

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, device_id=local_rank)

    rank = dist.get_rank()

    if enable_sp_ulysses:
        # Use SP Ulysses setup with default sizes
        return maybe_setup_distributed_sp_ulysses(sp_ulysses_size, sp_ring_size)

    logger.info(
        f"Started distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )
    return local_rank, world_size, rank, True


def maybe_destroy_distributed():
    """Destroys distributed process group if using distributed training."""
    if not is_distributed:
        # No distributed training
        return

    dist.destroy_process_group()
    logger.info(
        f"Destroyed distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )


def all_gather_tensor(
    local_tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[
                ctx.sp_rank
            ].contiguous(),
            None,
            None,
            None,
            None,
        )

def gather_outputs_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    if not group:
        group = get_draft_sp_group()
    if torch.distributed.get_world_size(group) == 1:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    return x