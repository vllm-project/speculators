import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from speculators.model import SpeculatorModel

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
is_distributed = "LOCAL_RANK" in os.environ

logger = logging.getLogger("speculators")


def maybe_setup_distributed() -> tuple[int, int, int, bool]:
    """Sets up distributed training if the process was launched with `torchrun`.
    If not, returns single process training.

    Based on of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

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

    logger.info(
        f"Started distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )
    return local_rank, world_size, rank, True


def maybe_destroy_distributed():
    """Destroys the distributed process group if using distributed training."""
    if not is_distributed:
        # No distributed training
        return

    dist.destroy_process_group()
    logger.info(
        f"Destroyed distributed with local_rank={local_rank}, world_size={world_size}",
        extra={"override_rank0_filter": True},
    )


def apply_fully_sharded(model: torch.nn.Module):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    """
    # Verify model is a registered SpeculatorModel
    if not isinstance(model, SpeculatorModel):
        raise TypeError(f"Model must be a SpeculatorModel, got {type(model).__name__}")

    model_class = type(model)
    registry = SpeculatorModel.registry
    if registry is None or model_class not in registry.values():
        raise ValueError(
            f"Model {model_class.__name__} is not registered in "
            f"SpeculatorModel.registry. "
            f"Available models: {list(registry.keys()) if registry else []}"
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for layer in model.layers:  # type: ignore[union-attr]
        # we apply fully_shard to each DecoderLayer
        layer.to_empty(device="meta")
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model)

    return model
