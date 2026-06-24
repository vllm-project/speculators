import logging
import warnings

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from speculators.data_generation.preprocessing import get_tokenizer, load_processor

logger = logging.getLogger("speculators")


def resolve_mask_token_id(
    verifier_name_or_path: str,
    vocab_size: int,
    mask_token_id: int | None = None,
    *,
    trust_remote_code: bool = False,
) -> int:
    """Resolve mask_token_id from explicit value, tokenizer, or fallback.

    Resolution order:
        1. Explicit mask_token_id if provided
        2. Tokenizer's existing mask_token_id
        3. Add <|MASK|> to tokenizer if embed_tokens has unused slots
        4. Fallback to pad/eos/unk token
    """
    if mask_token_id is not None:
        logger.info(f"Using explicit mask_token_id={mask_token_id}")
        return mask_token_id

    processor = load_processor(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = get_tokenizer(processor)

    if tokenizer.mask_token_id is not None:
        logger.info(f"Using tokenizer mask_token_id={tokenizer.mask_token_id}")
        return tokenizer.mask_token_id

    if len(tokenizer) < vocab_size:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        added_id: int = tokenizer.mask_token_id  # type: ignore[assignment]
        logger.warning(
            f"Added <|MASK|> to tokenizer, mask_token_id={added_id} "
            f"(tokenizer len={len(tokenizer)}, vocab_size={vocab_size})"
        )
        return added_id

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


def normalize_counted_metrics(
    metrics: dict[str, float], world_size: int = 1
) -> dict[str, float]:
    """Normalize metrics after ReduceOp.SUM across ranks.

    For any key ending in '_total', finds the matching '_sum' key,
    computes sum / total, and stores the result under the prefix
    (e.g. 'loss_sum' / 'loss_total' -> 'loss').
    The raw sum/total keys are removed.

    Any remaining metrics (not part of a sum/total pair) are divided
    by world_size to compute the average across ranks.
    """
    normalized_keys: set[str] = set()
    for tk in [k for k in metrics if k.endswith("_total")]:
        prefix = tk.removesuffix("_total")
        sk = f"{prefix}_sum"
        if sk in metrics:
            total = metrics[tk]
            metrics[prefix] = metrics[sk] / total if total > 0 else 0.0
            del metrics[sk]
            normalized_keys.add(prefix)
        del metrics[tk]

    if world_size > 1:
        for k in metrics:
            if k not in normalized_keys:
                metrics[k] /= world_size

    return metrics


def apply_fully_sharded(
    model: torch.nn.Module,
    process_group: dist.ProcessGroup | None = None,
):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function.

    Args:
        model: The model to shard.
        process_group: Optional process group for FSDP. When using sequence
            parallelism, pass the DP group so FSDP shards only across
            data-parallel ranks.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    fsdp_kwargs: dict = {"mp_policy": mp_policy}
    if process_group is not None:
        fsdp_kwargs["process_group"] = process_group

    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
