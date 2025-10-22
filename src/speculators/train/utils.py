import os

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_distributed = "LOCAL_RANK" in os.environ


def maybe_setup_distributed():
    # Based off of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
    if not is_distributed:
        # No distributed training
        return 0, 1, 0, False

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)

    rank = dist.get_rank()

    print(f"Started distributed with local_rank={local_rank}, world_size={world_size}")
    return local_rank, world_size, rank, True


def maybe_destroy_distributed():
    if not is_distributed:
        # No distributed training
        return

    dist.destroy_process_group()
    print(
        f"Destroyed distributed with local_rank={local_rank}, world_size={world_size}"
    )
