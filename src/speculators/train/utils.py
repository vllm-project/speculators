import os
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))


def log_rank0(message):
    if local_rank != 0:
        return
    print(message)


def maybe_setup_distributed():
    # Based off of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
    if "LOCAL_RANK" not in os.environ:
        # No distributed training
        return 0, 1, 0, False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)

    rank = dist.get_rank()

    print(
        f"Started DDP with local_rank={local_rank}, world_size={world_size}, rank={rank}"
    )
    return local_rank, world_size, rank, True


def maybe_destroy_distributed():
    if "LOCAL_RANK" not in os.environ:
        # No distributed training
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = dist.get_rank()

    dist.destroy_process_group()
    print(
        f"Destroyed DDP with local_rank={local_rank}, world_size={world_size}, rank={rank}"
    )
