"""MTP weight stitcher — merges finetuned MTP head back into a verifier.

After finetuning an MTP head with speculators, this module replaces the
original MTP weights in the verifier checkpoint with the finetuned
weights. Non-MTP shards are copied as-is, so the output is a complete,
deployable model directory.
"""

import json
import re
import shutil
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from speculators.convert.eagle.utils import ensure_checkpoint_is_local
from speculators.convert.mtp.converter import EXACT_KEY_MAP, PREFIX_KEY_MAP
from speculators.models.mtp import MTPDraftModel

__all__ = ["MTPStitcher"]

_FROZEN_KEYS = {"embed_tokens.weight", "lm_head.weight"}

_EXACT_INVERSE: dict[str, str] = {v: k for k, v in EXACT_KEY_MAP.items()}

_PREFIX_INVERSE: list[tuple[str, str]] = [(dst, src) for src, dst in PREFIX_KEY_MAP]

_FUSED_EXPERT_PATTERN = re.compile(r"^(.+\.experts)\.(gate_up_proj|down_proj)$")


class MTPStitcher:
    """Merge a finetuned MTP checkpoint back into a verifier.

    Remaps speculators-format keys (``mtp_layers.0.*``) back to the
    native format (``mtp.*``) and replaces the corresponding tensors in
    the verifier's sharded safetensors files. Frozen weights
    (``embed_tokens``, ``lm_head``) are skipped — the verifier's
    originals are preserved.
    """

    def stitch(
        self,
        finetuned_checkpoint: str | Path,
        verifier_path: str | Path,
        output_path: str | Path,
        validate: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Stitch finetuned MTP weights into a verifier checkpoint.

        :param finetuned_checkpoint: Path to the finetuned speculators
            checkpoint directory (model.safetensors + config.json).
        :param verifier_path: Local path or HF model ID for the
            verifier.
        :param output_path: Directory for the stitched model.
        :param validate: Verify the output can be loaded by
            safetensors.
        :param cache_dir: Optional cache directory for Hub downloads.
        """
        logger.info(f"Stitching finetuned MTP head from: {finetuned_checkpoint}")

        remapped = self._load_and_remap(str(finetuned_checkpoint))
        logger.info(f"Remapped {len(remapped)} MTP weight tensors")

        verifier_local = ensure_checkpoint_is_local(verifier_path, cache_dir)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        weight_map = self._load_weight_map(verifier_local)
        self._process_shards(verifier_local, output_dir, weight_map, remapped)
        self._copy_metadata(verifier_local, output_dir)

        logger.success(f"Stitched model saved to: {output_dir}")

        if validate:
            self._validate(output_dir, list(remapped.keys()))

    def _load_and_remap(self, checkpoint_path: str) -> dict[str, torch.Tensor]:
        model = MTPDraftModel.from_pretrained(checkpoint_path)

        remapped: dict[str, torch.Tensor] = {}
        for key, tensor in model.state_dict().items():
            new_key = _remap_key_to_source(key)
            if new_key is not None:
                remapped[new_key] = tensor
        return _unfuse_moe_experts(remapped)

    def _load_weight_map(self, verifier_local: Path) -> dict[str, str]:
        index_path = verifier_local / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Expected sharded checkpoint at {verifier_local} "
                "(model.safetensors.index.json not found)"
            )
        with index_path.open() as f:
            return json.load(f)["weight_map"]

    def _process_shards(
        self,
        verifier_local: Path,
        output_dir: Path,
        weight_map: dict[str, str],
        remapped: dict[str, torch.Tensor],
    ) -> None:
        mtp_shards = {weight_map[k] for k in remapped if k in weight_map}

        for shard_name in sorted(set(weight_map.values())):
            src = verifier_local / shard_name
            dst = output_dir / shard_name

            if shard_name not in mtp_shards:
                shutil.copy2(src, dst)
                continue

            with safe_open(str(src), framework="pt") as f:
                shard_data = {
                    k: f.get_tensor(k)
                    for k in f.keys()  # noqa: SIM118
                }

            replaced = []
            for mtp_key, tensor in remapped.items():
                if weight_map.get(mtp_key) == shard_name:
                    shard_data[mtp_key] = tensor
                    replaced.append(mtp_key)

            save_file(shard_data, str(dst))
            logger.info(f"Replaced {len(replaced)} MTP tensor(s) in {shard_name}")

    def _copy_metadata(self, verifier_local: Path, output_dir: Path) -> None:
        index_path = verifier_local / "model.safetensors.index.json"
        shutil.copy2(
            index_path,
            output_dir / "model.safetensors.index.json",
        )
        for pattern in (
            "config*.json",
            "tokenizer*",
            "*.tiktoken",
        ):
            for f in verifier_local.glob(pattern):
                shutil.copy2(f, output_dir / f.name)

    def _validate(self, output_dir: Path, expected_keys: list[str]) -> None:
        logger.info("Validating stitched checkpoint...")
        index_path = output_dir / "model.safetensors.index.json"
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]

        for key in expected_keys:
            if key not in weight_map:
                logger.warning(f"Expected key {key} not in output weight map")
                continue
            shard = weight_map[key]
            shard_path = output_dir / shard
            with safe_open(str(shard_path), framework="pt") as f:
                if key not in f.keys():  # noqa: SIM118
                    raise ValueError(
                        f"Key {key} listed in index but missing from shard {shard}"
                    )

        logger.success("Validation succeeded")


def _remap_key_to_source(key: str) -> str | None:
    """Map a speculators key back to native format.

    Returns None for frozen weights that should be skipped.
    """
    if key in _FROZEN_KEYS:
        return None
    if key in _EXACT_INVERSE:
        return _EXACT_INVERSE[key]
    for src, dst in _PREFIX_INVERSE:
        if key.startswith(src):
            return dst + key[len(src) :]
    logger.warning(f"Unknown key '{key}' has no remapping, skipping")
    return None


def _unfuse_moe_experts(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expand packed expert tensors back to individual weights.

    Inverse of ``MTPConverter._fuse_moe_experts``.  MoE checkpoints
    (e.g. Qwen3-Next) expect per-expert keys (``experts.{N}.gate_proj``),
    so fused 3D tensors from the model's state_dict must be split back.
    """
    result: dict[str, torch.Tensor] = {}
    for key, tensor in weights.items():
        m = _FUSED_EXPERT_PATTERN.match(key)
        if not m:
            result[key] = tensor
            continue

        prefix, proj = m.group(1), m.group(2)
        _expected_expert_ndim = 3
        if tensor.ndim != _expected_expert_ndim:
            raise ValueError(
                f"Expected 3D expert tensor for {key}, got shape {tensor.shape}"
            )
        num_experts = tensor.shape[0]
        if proj == "gate_up_proj":
            for i in range(num_experts):
                gate, up = tensor[i].chunk(2, dim=0)
                result[f"{prefix}.{i}.gate_proj.weight"] = gate
                result[f"{prefix}.{i}.up_proj.weight"] = up
        else:
            for i in range(num_experts):
                result[f"{prefix}.{i}.down_proj.weight"] = tensor[i]
        logger.debug(f"Unfused {num_experts} experts at {prefix}.{proj}")

    return result
