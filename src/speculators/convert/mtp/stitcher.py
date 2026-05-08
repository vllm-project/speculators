"""MTP checkpoint stitcher.

Takes finetuned MTP weights in Speculators format and stitches them back
into a full verifier checkpoint, producing a self-contained checkpoint
directory that vLLM can load directly.

Only the ``mtp_layers.*`` subtree is read from the finetuned checkpoint;
embed_tokens and lm_head are frozen copies of the verifier's weights and
are skipped.
"""

import json
import re
import shutil
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

__all__ = ["MTPStitcher"]

INVERSE_EXACT_KEY_MAP: dict[str, str] = {
    "mtp_layers.0.input_proj.weight": "mtp.fc.weight",
    "mtp_layers.0.final_layernorm.weight": "mtp.norm.weight",
}

INVERSE_PREFIX_KEY_MAP: list[tuple[str, str]] = [
    ("mtp_layers.0.hidden_layernorm.", "mtp.pre_fc_norm_hidden."),
    ("mtp_layers.0.token_layernorm.", "mtp.pre_fc_norm_embedding."),
    ("mtp_layers.0.", "mtp.layers.0."),
]

_FROZEN_KEYS = {"embed_tokens.weight", "lm_head.weight"}

_FUSED_GATE_UP_PATTERN = re.compile(r"^(.+\.experts)\.gate_up_proj$")
_FUSED_DOWN_PATTERN = re.compile(r"^(.+\.experts)\.down_proj$")


class MTPStitcher:
    """Reintegrate finetuned MTP weights into a full verifier checkpoint.

    Produces a complete, self-contained checkpoint directory that is
    directly deployable on vLLM and uploadable to HF Hub.
    """

    @staticmethod
    def _remap_key(key: str) -> str:
        """Remap a speculators-format key to native verifier format."""
        if key in INVERSE_EXACT_KEY_MAP:
            return INVERSE_EXACT_KEY_MAP[key]
        for src, dst in INVERSE_PREFIX_KEY_MAP:
            if key.startswith(src):
                return dst + key[len(src) :]
        return key

    @staticmethod
    def _filter_frozen_keys(
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remove frozen weights that should not be stitched."""
        return {k: v for k, v in weights.items() if k not in _FROZEN_KEYS}

    @staticmethod
    def _unfuse_moe_experts(
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Unfuse packed 3D expert tensors back to per-expert format.

        Inverse of ``MTPConverter._fuse_moe_experts``.
        """
        result: dict[str, torch.Tensor] = {}
        gate_up_keys: dict[str, torch.Tensor] = {}
        down_keys: dict[str, torch.Tensor] = {}

        for key, tensor in weights.items():
            m_gu = _FUSED_GATE_UP_PATTERN.match(key)
            m_d = _FUSED_DOWN_PATTERN.match(key)
            if m_gu:
                gate_up_keys[m_gu.group(1)] = tensor
            elif m_d:
                down_keys[m_d.group(1)] = tensor
            else:
                result[key] = tensor

        if not gate_up_keys:
            return weights

        for prefix, gate_up in gate_up_keys.items():
            down = down_keys[prefix]

            num_experts = gate_up.shape[0]
            half = gate_up.shape[1] // 2

            for i in range(num_experts):
                result[f"{prefix}.{i}.gate_proj.weight"] = gate_up[
                    i, :half
                ].contiguous()
                result[f"{prefix}.{i}.up_proj.weight"] = gate_up[i, half:].contiguous()
                result[f"{prefix}.{i}.down_proj.weight"] = down[i].contiguous()

            logger.debug(
                f"Unfused {num_experts} experts at {prefix}: "
                f"gate_up_proj {gate_up.shape} -> per-expert"
            )

        return result

    def _load_finetuned_weights(self, checkpoint_dir: Path) -> dict[str, torch.Tensor]:
        """Load all tensors from a speculators-format checkpoint."""
        weights: dict[str, torch.Tensor] = {}

        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as f:
                weight_map = json.load(f)["weight_map"]
            for shard in set(weight_map.values()):
                with safe_open(str(checkpoint_dir / shard), framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        weights[key] = f.get_tensor(key)
            return weights

        single = checkpoint_dir / "model.safetensors"
        if single.exists():
            with safe_open(str(single), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    weights[key] = f.get_tensor(key)
            return weights

        raise FileNotFoundError(f"No safetensors found at {checkpoint_dir}")

    @staticmethod
    def _copy_checkpoint(src: Path, dst: Path) -> None:
        """Copy verifier checkpoint directory to output location."""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _stitch_sharded(
        self,
        output_dir: Path,
        native_weights: dict[str, torch.Tensor],
    ) -> None:
        """Replace MTP weights in a sharded checkpoint."""
        index_path = output_dir / "model.safetensors.index.json"
        with index_path.open() as f:
            index_data = json.load(f)
        weight_map: dict[str, str] = index_data["weight_map"]

        shard_to_new: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in native_weights.items():
            shard = weight_map.get(key)
            if shard is None:
                raise ValueError(
                    f"Finetuned key '{key}' not found in verifier weight map. "
                    "The finetuned checkpoint may not match the verifier."
                )
            shard_to_new.setdefault(shard, {})[key] = tensor

        for shard_filename, new_weights in shard_to_new.items():
            shard_path = output_dir / shard_filename
            existing: dict[str, torch.Tensor] = {}
            metadata = None
            with safe_open(str(shard_path), framework="pt") as f:
                metadata = f.metadata()
                for k in f.keys():  # noqa: SIM118
                    existing[k] = f.get_tensor(k)
            existing.update(new_weights)
            save_file(existing, str(shard_path), metadata=metadata)
            logger.debug(f"Updated {len(new_weights)} key(s) in shard {shard_filename}")

    def _stitch_single(
        self,
        safetensors_path: Path,
        native_weights: dict[str, torch.Tensor],
    ) -> None:
        """Replace MTP weights in a single-file checkpoint."""
        existing: dict[str, torch.Tensor] = {}
        metadata = None
        with safe_open(str(safetensors_path), framework="pt") as f:
            metadata = f.metadata()
            for k in f.keys():  # noqa: SIM118
                existing[k] = f.get_tensor(k)
        existing.update(native_weights)
        save_file(existing, str(safetensors_path), metadata=metadata)

    def stitch(
        self,
        finetuned_checkpoint: str | Path,
        verifier_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Stitch finetuned MTP weights back into a verifier checkpoint.

        Args:
            finetuned_checkpoint: Path to the finetuned speculators checkpoint.
            verifier_path: Path to the original verifier checkpoint.
            output_path: Path for the output checkpoint directory.

        Returns:
            The output path.
        """
        finetuned_checkpoint = Path(finetuned_checkpoint)
        verifier_path = Path(verifier_path)
        output_path = Path(output_path)

        logger.info(
            f"Stitching MTP checkpoint: {finetuned_checkpoint} -> {output_path}"
        )

        weights = self._load_finetuned_weights(finetuned_checkpoint)
        logger.info(f"Loaded {len(weights)} finetuned weight tensors")

        weights = self._filter_frozen_keys(weights)
        weights = self._unfuse_moe_experts(weights)
        native_weights = {self._remap_key(k): v for k, v in weights.items()}
        logger.info(f"Remapped {len(native_weights)} keys to native format")

        self._copy_checkpoint(verifier_path, output_path)

        index_path = output_path / "model.safetensors.index.json"
        if index_path.exists():
            self._stitch_sharded(output_path, native_weights)
        else:
            single = output_path / "model.safetensors"
            if single.exists():
                self._stitch_single(single, native_weights)
            else:
                raise FileNotFoundError(
                    f"No safetensors checkpoint found at {output_path}"
                )

        logger.success(f"Stitched checkpoint saved to: {output_path}")
        return output_path
