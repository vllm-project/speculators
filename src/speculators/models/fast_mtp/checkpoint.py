"""Key remapping between speculators checkpoint format and Qwen3-Next vLLM format.

speculators training uses PyTorch-native state-dict keys::

    mtp_layers.0.{suffix}

vLLM expects the original Qwen3-Next keys::

    mtp.{suffix}          (for the 4 FastMTP mixin modules)
    mtp.layers.0.{suffix} (for all other MTP weights: attention, MLP)

The 4 :class:`~speculators.models.fast_mtp.model_definitions.FastMTPLayerMixin`
module attributes (``pre_fc_norm_hidden``, ``pre_fc_norm_embedding``, ``fc``,
``norm``) are named to match Qwen3-Next's original weight keys exactly.  Because
attribute names align with Qwen3-Next names, the entire remapping logic reduces to a
single frozenset membership check — no explicit dict is required.

This module is the single source of truth for key remapping and is used by both
``examples/fast_mtp/stitch_weights.py`` (speculators → Qwen3-Next) and
``examples/fast_mtp/convert_checkpoint.py`` (Qwen3-Next → speculators).
"""

import json
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

__all__ = [
    "SKIP_KEYS",
    "remap_key",
    "remap_keys",
    "update_weight_index",
]

# Keys present in the speculator checkpoint that belong to the verifier (frozen
# weights copied in from the verifier during training).  These are omitted from
# the stitched model because the verifier shards already contain them.
SKIP_KEYS: frozenset[str] = frozenset({"embed_tokens.weight", "lm_head.weight"})

# The ``_SPECULATORS_PREFIX`` is the state-dict namespace used by the single-element
# ``mtp_layers`` ModuleList during training.  The format is chosen so that
# ``remap_key`` can strip the prefix and determine the Qwen3-Next namespace purely
# from whether the remaining suffix is a top-level mixin module weight.
_SPECULATORS_PREFIX = "mtp_layers.0."

# These four weight keys live directly under ``mtp.`` in the Qwen3-Next checkpoint
# (not under ``mtp.layers.0.``).  They correspond to the four
# :class:`FastMTPLayerMixin` module attributes whose names match Qwen3-Next exactly.
# All other MTP weights (attention, MLP) live under ``mtp.layers.0.``.
_MTP_TOP_LEVEL_SUFFIXES: frozenset[str] = frozenset(
    {
        "pre_fc_norm_hidden.weight",
        "pre_fc_norm_embedding.weight",
        "fc.weight",
        "norm.weight",
    }
)


def remap_key(key: str) -> str:
    """Map a speculators checkpoint key to the original Qwen3-Next ``mtp.*`` format.

    :param key: A state-dict key starting with ``mtp_layers.0.``.
    :returns: The corresponding Qwen3-Next key (``mtp.{suffix}`` for mixin
        modules, ``mtp.layers.0.{suffix}`` for attention/MLP weights).
    :raises ValueError: If ``key`` does not start with ``_SPECULATORS_PREFIX``.
    """
    if not key.startswith(_SPECULATORS_PREFIX):
        raise ValueError(
            f"Cannot remap key {key!r}: expected prefix {_SPECULATORS_PREFIX!r}"
        )
    suffix = key[len(_SPECULATORS_PREFIX) :]
    if suffix in _MTP_TOP_LEVEL_SUFFIXES:
        return f"mtp.{suffix}"
    return f"mtp.layers.0.{suffix}"


def remap_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a speculators state dict to Qwen3-Next ``mtp.*`` format.

    Skips keys in :data:`SKIP_KEYS` (frozen verifier weights).

    :param state_dict: State dict from a trained FastMTP checkpoint.
    :returns: Remapped dict with Qwen3-Next-compatible keys.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if any(key.startswith(skip) for skip in SKIP_KEYS):
            continue
        remapped[remap_key(key)] = tensor
    return remapped


def update_weight_index(
    verifier_dir: Path,
    output_dir: Path,
    mtp_keys: list[str],
    new_shard_name: str,
) -> None:
    """Update the verifier's safetensors index to route ``mtp.*`` keys to a new shard.

    Removes all existing ``mtp.*`` entries from the original index (so the
    finetuned weights override them) and adds new entries for all remapped MTP
    keys, pointing to ``new_shard_name``.

    :param verifier_dir: Directory containing the original verifier checkpoint,
        including ``model.safetensors.index.json``.
    :param output_dir: Directory where the updated index will be written.
    :param mtp_keys: Remapped MTP key names (in Qwen3-Next format) to add to
        the index pointing at ``new_shard_name``.
    :param new_shard_name: Filename of the new finetuned MTP shard
        (e.g. ``mtp_finetuned.safetensors``).
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

    # Remove stale mtp.* entries from the original index so our finetuned
    # weights are the sole source for all MTP keys.
    removed = [k for k in list(weight_map.keys()) if k.startswith("mtp.")]
    for key in removed:
        del weight_map[key]
    log.info("  Removed %d original mtp.* entries from index", len(removed))

    # Add new MTP entries pointing to our finetuned shard
    for key in mtp_keys:
        weight_map[key] = new_shard_name

    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2)
