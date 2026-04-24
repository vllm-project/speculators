"""MTP checkpoint converter.

Extracts the MTP layer (plus embed_tokens and lm_head) from a checkpoint
with native MTP layers and saves a self-contained Speculators checkpoint
that loads with ``MTPDraftModel.from_pretrained(path)``.

Only the ``mtp.*`` subtree — plus ``embed_tokens`` and ``lm_head`` — is
read from the (potentially sharded) safetensors file; the rest of the
model is never loaded.
"""

import json
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
)
from speculators.models.mtp import MTPConfig, MTPDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["MTPConverter"]

_MTP_PREFIX = "mtp."

_EMBED_KEYS = {
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
}
_LM_HEAD_KEY = "lm_head.weight"

EXACT_KEY_MAP: dict[str, str] = {
    "mtp.fc.weight": "mtp_layers.0.input_proj.weight",
    "mtp.norm.weight": "mtp_layers.0.final_layernorm.weight",
}

PREFIX_KEY_MAP: list[tuple[str, str]] = [
    ("mtp.pre_fc_norm_hidden.", "mtp_layers.0.hidden_layernorm."),
    ("mtp.pre_fc_norm_embedding.", "mtp_layers.0.token_layernorm."),
    ("mtp.layers.0.", "mtp_layers.0."),
]

_EXPERT_PATTERN = re.compile(
    r"^(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


class MTPConverter:
    """Extract the MTP head from a checkpoint with native MTP layers.

    Reads only the MTP layer, embed_tokens, and lm_head from the source
    checkpoint.  Sharded safetensors files are handled transparently via
    the weight index — the main transformer stack is never loaded.
    """

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        num_speculative_steps: int = 3,
        validate: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting MTP checkpoint: {input_path}")

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = load_checkpoint_config(local_path)

        all_keys = self._list_checkpoint_keys(local_path)
        self._verify_mtp_format(all_keys)

        weights = self._extract_weights(local_path, all_keys)
        weights = self._fuse_moe_experts(weights)
        logger.info(f"Extracted {len(weights)} weight tensors")

        if "text_config" in source_config:
            source_config = source_config["text_config"]

        config = self._build_config(source_config, base_model, num_speculative_steps)
        saved_path = self._save(config, weights, output_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate(saved_path)

    def _list_checkpoint_keys(self, checkpoint_dir: Path) -> list[str]:
        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as f:
                return list(json.load(f)["weight_map"].keys())

        single = checkpoint_dir / "model.safetensors"
        if single.exists():
            with safe_open(str(single), framework="pt") as f:
                return list(f.keys())  # noqa: SIM118

        pytorch = checkpoint_dir / "pytorch_model.bin"
        if pytorch.exists():
            state_dict = torch.load(str(pytorch), map_location="cpu", weights_only=True)
            return list(state_dict.keys())

        raise FileNotFoundError(f"No checkpoint weights found at {checkpoint_dir}")

    def _verify_mtp_format(self, keys: list[str]) -> None:
        if not any(k.startswith(_MTP_PREFIX) for k in keys):
            raise ValueError(
                f"No keys with prefix '{_MTP_PREFIX}' found. "
                "This converter requires checkpoints with native "
                "MTP layers (e.g. Qwen3-Next, Qwen3.5, Qwen3.5-MoE)."
            )

    @staticmethod
    def _remap_key(key: str) -> str:
        if key in _EMBED_KEYS:
            return "embed_tokens.weight"
        if key in EXACT_KEY_MAP:
            return EXACT_KEY_MAP[key]
        for src, dst in PREFIX_KEY_MAP:
            if key.startswith(src):
                return dst + key[len(src) :]
        return key

    @staticmethod
    def _fuse_moe_experts(
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse individual expert weights into packed tensors.

        This exists to support transformers>=4.57. When the minimum is
        raised to v5+, replace with ``WeightConverter`` from
        ``transformers.core_model_loading`` (MergeModulelist + Concatenate).

        MoE checkpoints (e.g. Qwen3-Next) store per-expert weights as
        separate keys (``experts.{N}.gate_proj``), but the transformers
        model uses fused 3D tensors (``experts.gate_up_proj``).  This
        bridges the checkpoint→model format gap.

        ``experts.{N}.gate_proj`` + ``experts.{N}.up_proj`` ->
        ``experts.gate_up_proj`` and ``experts.{N}.down_proj`` ->
        ``experts.down_proj`` (stacked along dim 0).
        """
        groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        non_expert: dict[str, torch.Tensor] = {}

        for key, tensor in weights.items():
            m = _EXPERT_PATTERN.match(key)
            if m:
                prefix = m.group(1)
                idx = int(m.group(2))
                proj = m.group(3)
                groups.setdefault(prefix, {}).setdefault(idx, {})[proj] = tensor
            else:
                non_expert[key] = tensor

        if not groups:
            return weights

        for prefix, experts_by_idx in groups.items():
            num_experts = max(experts_by_idx.keys()) + 1
            expected = set(range(num_experts))
            if set(experts_by_idx.keys()) != expected:
                raise ValueError(
                    f"Non-contiguous expert indices at {prefix}: "
                    f"found {sorted(experts_by_idx.keys())}, "
                    f"expected 0..{num_experts - 1}"
                )

            for i in range(num_experts):
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    if proj_name not in experts_by_idx[i]:
                        raise ValueError(
                            f"Expert {i} at {prefix} missing {proj_name} weight"
                        )

            gate_list = [experts_by_idx[i]["gate_proj"] for i in range(num_experts)]
            up_list = [experts_by_idx[i]["up_proj"] for i in range(num_experts)]
            down_list = [experts_by_idx[i]["down_proj"] for i in range(num_experts)]

            gate_up = torch.stack(
                [
                    torch.cat([g, u], dim=0)
                    for g, u in zip(gate_list, up_list, strict=True)
                ]
            )
            down = torch.stack(down_list)

            non_expert[f"{prefix}.gate_up_proj"] = gate_up
            non_expert[f"{prefix}.down_proj"] = down
            logger.debug(
                f"Fused {num_experts} experts at {prefix}: "
                f"gate_up_proj={gate_up.shape}, "
                f"down_proj={down.shape}"
            )

        return non_expert

    def _extract_weights(
        self, checkpoint_dir: Path, all_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        needed = {
            k
            for k in all_keys
            if k in _EMBED_KEYS or k == _LM_HEAD_KEY or k.startswith(_MTP_PREFIX)
        }

        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            return self._extract_from_shards(checkpoint_dir, index_path, needed)

        single = checkpoint_dir / "model.safetensors"
        if single.exists():
            weights: dict[str, torch.Tensor] = {}
            with safe_open(str(single), framework="pt") as f:
                for key in needed & set(f.keys()):  # noqa: SIM118
                    weights[self._remap_key(key)] = f.get_tensor(key)
            return weights

        pytorch = checkpoint_dir / "pytorch_model.bin"
        if pytorch.exists():
            state_dict = torch.load(str(pytorch), map_location="cpu", weights_only=True)
            return {self._remap_key(k): v for k, v in state_dict.items() if k in needed}

        raise FileNotFoundError(f"No checkpoint weights found at {checkpoint_dir}")

    def _extract_from_shards(
        self,
        checkpoint_dir: Path,
        index_path: Path,
        needed_keys: set[str],
    ) -> dict[str, torch.Tensor]:
        with index_path.open() as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        shard_to_keys: dict[str, list[str]] = {}
        for key in needed_keys:
            if shard := weight_map.get(key):
                shard_to_keys.setdefault(shard, []).append(key)

        weights: dict[str, torch.Tensor] = {}
        for shard_filename, keys in shard_to_keys.items():
            shard_path = checkpoint_dir / shard_filename
            logger.debug(f"Reading {len(keys)} key(s) from shard {shard_filename}")
            with safe_open(str(shard_path), framework="pt") as f:
                for key in keys:
                    weights[self._remap_key(key)] = f.get_tensor(key)

        return weights

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        num_speculative_steps: int,
    ) -> MTPConfig:
        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        speculators_config = SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=num_speculative_steps)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=base_model,
                architectures=verifier_config_dict.get("architectures", []),
            ),
        )
        return MTPConfig(
            transformer_layer_config=source_config,  # type: ignore[arg-type]
            speculators_config=speculators_config,
        )

    def _save(
        self,
        config: MTPConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = MTPDraftModel(config)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(
                "Unexpected keys in extracted weights — the "
                "checkpoint structure does not match the model "
                "architecture. This may indicate a wrong layer_idx "
                "(full_attention vs linear_attention). "
                f"Unexpected keys: {unexpected}"
            )
        if missing:
            logger.debug(f"Keys not in extracted weights (filled by init): {missing}")

        float_dtypes = {t.dtype for t in weights.values() if t.is_floating_point()}
        if float_dtypes:
            if len(float_dtypes) > 1:
                logger.warning(
                    f"Mixed float dtypes in checkpoint: {float_dtypes}. "
                    "Using first encountered dtype."
                )
            model.to(dtype=next(iter(float_dtypes)))  # type: ignore[call-arg]

        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        logger.info("Validating converted checkpoint...")
        try:
            MTPDraftModel.from_pretrained(str(output_path))
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
