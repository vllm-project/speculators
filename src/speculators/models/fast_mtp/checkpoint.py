"""Checkpoint utilities for FastMTP.

The ``MTPBlock`` module layout mirrors the Qwen3-Next checkpoint structure
directly, so training state-dict keys (``mtp.*``) already match what vLLM
expects.  No key remapping is required between training, conversion, and
inference — only the frozen verifier weights (``embed_tokens.weight``,
``lm_head.weight``) need to be filtered out before stitching, because those
are already present in the verifier shards.
"""

import json
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

__all__ = [
    "SKIP_KEYS",
    "filter_mtp_keys",
    "update_weight_index",
]

# Keys present in the speculator checkpoint that belong to the verifier (frozen
# weights loaded during training).  These are omitted from the stitched model
# because the verifier shards already contain them.
SKIP_KEYS: frozenset[str] = frozenset({"embed_tokens.weight", "lm_head.weight"})


def filter_mtp_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return MTP weights from *state_dict*, excluding frozen verifier weights.

    Training checkpoints contain ``embed_tokens.weight`` and ``lm_head.weight``
    (loaded from the verifier for the forward pass) but these are already
    present in the verifier shards of the stitched model.  All other keys are
    returned unchanged — no renaming is needed because the ``MTPBlock`` module
    layout mirrors the original Qwen3-Next checkpoint structure directly.

    :param state_dict: State dict from a trained FastMTP checkpoint.
    :returns: Filtered dict with verifier weights removed.
    """
    return {k: v for k, v in state_dict.items() if k not in SKIP_KEYS}


def update_weight_index(
    verifier_dir: Path,
    output_dir: Path,
    mtp_keys: list[str],
    new_shard_name: str,
) -> None:
    """Update the verifier's safetensors index to route ``mtp.*`` keys to a new shard.

    Removes all existing ``mtp.*`` entries from the original index (so the
    finetuned weights override them) and adds new entries for all MTP keys,
    pointing to ``new_shard_name``.

    :param verifier_dir: Directory containing the original verifier checkpoint,
        including ``model.safetensors.index.json``.
    :param output_dir: Directory where the updated index will be written.
    :param mtp_keys: MTP key names to add to the index pointing at ``new_shard_name``.
    :param new_shard_name: Filename of the new finetuned MTP shard.
    :raises FileNotFoundError: If ``model.safetensors.index.json`` is absent
        from ``verifier_dir`` (single-shard models are not supported).
    """
    index_path = verifier_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No model.safetensors.index.json found in {verifier_dir}. "
            "Single-shard verifier checkpoints are not supported by stitch_weights."
        )

    with index_path.open() as f:
        index = json.load(f)

    weight_map: dict[str, str] = index.get("weight_map", {})

    removed = [k for k in list(weight_map.keys()) if k.startswith("mtp.")]
    for key in removed:
        del weight_map[key]
    log.info("  Removed %d original mtp.* entries from index", len(removed))

    for key in mtp_keys:
        weight_map[key] = new_shard_name

    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2)
