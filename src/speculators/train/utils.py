import argparse
import logging
import os
import warnings

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer

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


def resolve_mask_token_id(args: argparse.Namespace, vocab_size: int) -> int:
    """Resolve mask_token_id from args, tokenizer, or fallback.

    Resolution order:
        1. Explicit --mask-token-id argument
        2. Tokenizer's existing mask_token_id
        3. Add <|MASK|> to tokenizer if embed_tokens has unused slots
        4. Fallback to pad/eos/unk token
    """
    if args.mask_token_id is not None:
        logger.info(f"Using explicit mask_token_id={args.mask_token_id}")
        return args.mask_token_id

    tokenizer = AutoTokenizer.from_pretrained(args.verifier_name_or_path)

    if tokenizer.mask_token_id is not None:
        logger.info(f"Using tokenizer mask_token_id={tokenizer.mask_token_id}")
        return tokenizer.mask_token_id

    if len(tokenizer) < vocab_size:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
        assert mask_token_id is not None
        logger.info(
            f"Added <|MASK|> to tokenizer, mask_token_id={mask_token_id} "
            f"(tokenizer len={len(tokenizer)}, vocab_size={vocab_size})"
        )
        return mask_token_id

    for token_name in ("pad_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, token_name, None)
        if token_id is not None:
            warnings.warn(
                f"Tokenizer does not have mask_token and no unused embedding slots. "
                f"Using {token_name}={token_id} as fallback.",
                stacklevel=2,
            )
            return token_id

    raise ValueError(
        "Could not resolve mask_token_id: no --mask-token-id provided, tokenizer has "
        "no mask_token, no unused embedding slots, and no pad/eos/unk fallback tokens."
    )


def apply_fully_sharded(model: torch.nn.Module):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model)

    return model
