import datetime
import importlib.metadata
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

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


def save_train_command(save_path: str) -> None:
    """Write the launch command and provenance header to save_path/train_command.txt."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"

    pkg_versions: list[str] = []
    for pkg in ("speculators", "vllm", "transformers", "torch", "compressed-tensors"):
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        pkg_versions.append(f"# {pkg}: {ver}")

    header = "\n".join(
        [
            f"# Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            f"# Git SHA: {sha}",
            f"# World size: {os.environ.get('WORLD_SIZE', '1')}",
            *pkg_versions,
        ]
    )

    command = shlex.join(sys.argv)
    content = f"{header}\n{command}\n"

    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=save_path, prefix=".train_command_", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        tmp_path.replace(path / "train_command.txt")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def apply_fully_sharded(model: torch.nn.Module):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            # Skip buffer-only modules (no trainable params)
            fully_shard(module, mp_policy=mp_policy)
        elif isinstance(module, torch.nn.ModuleList):
            for child in module:
                if len(list(child.parameters())) > 0:
                    fully_shard(child, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
