"""FastMTP checkpoint converter.

Extracts the MTP layer (plus embeddings and LM head) from a full Qwen3-Next or
TencentBAC/MiMo checkpoint and saves it as a self-contained Speculators checkpoint.

Two source formats are supported:
- ``native_hf``: Hugging Face Qwen3-Next; MTP keys start with ``model.mtp_layers.``
- ``tencentbac``: MiMo / TencentBAC format; MTP keys start with ``mtp.``

The output checkpoint is self-contained: ``embed_tokens.weight`` and
``lm_head.weight`` are included so ``FastMTPSpeculator.from_pretrained(path)``
works without a verifier path.
"""

import json
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from safetensors import safe_open
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
)
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["FastMTPConverter"]

_Format = Literal["native_hf", "tencentbac"]

# Embed and LM-head keys that may appear in either source format.
# native_hf always uses the "model." prefix; tencentbac may omit it.
_NATIVE_HF_EMBED_KEYS: frozenset[str] = frozenset({"model.embed_tokens.weight"})
_NATIVE_HF_LM_HEAD_KEYS: frozenset[str] = frozenset({"model.lm_head.weight"})
_TENCENTBAC_EMBED_KEYS: frozenset[str] = frozenset(
    {"model.embed_tokens.weight", "embed_tokens.weight"}
)
_TENCENTBAC_LM_HEAD_KEYS: frozenset[str] = frozenset(
    {"model.lm_head.weight", "lm_head.weight"}
)


class FastMTPConverter:
    """Convert FastMTP checkpoints to the Speculators format.

    Supports two source formats:

    - **native_hf** (Qwen3-Next): keys prefixed with ``model.mtp_layers.``
    - **tencentbac** (MiMo): keys prefixed with ``mtp.``

    Sharded safetensors checkpoints are handled by streaming only the relevant
    shards — the full model is never loaded into memory.
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
        """Convert a FastMTP checkpoint to the Speculators format.

        :param input_path: Local path or Hugging Face model ID for the source.
        :param output_path: Directory where the converted checkpoint will be written.
        :param base_model: Verifier model ID or path; stored in VerifierConfig.
        :param num_speculative_steps: Number of MTP steps to configure in the output.
        :param validate: Load the output checkpoint to check for missing keys.
        :param cache_dir: Optional cache directory for Hub downloads.
        """
        logger.info(f"Converting FastMTP checkpoint: {input_path}")

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = load_checkpoint_config(local_path)

        all_keys = self._list_checkpoint_keys(local_path)
        fmt = self._detect_format(all_keys)
        logger.info(f"Detected checkpoint format: {fmt}")

        weights = self._extract_weights(local_path, all_keys, fmt)
        logger.info(f"Extracted {len(weights)} weight tensors")

        config = self._build_config(source_config, base_model, num_speculative_steps)

        saved_path = self._save(config, weights, output_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate(saved_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_checkpoint_keys(self, checkpoint_dir: Path) -> list[str]:
        """Return all weight key names without loading tensor data."""
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

    def _detect_format(self, keys: list[str]) -> _Format:
        """Identify whether the checkpoint uses native_hf or tencentbac key layout."""
        for key in keys:
            if key.startswith("model.mtp_layers."):
                return "native_hf"
            if key.startswith("mtp."):
                return "tencentbac"
        sample = keys[:20]
        raise ValueError(
            "Cannot detect FastMTP checkpoint format — no keys starting with "
            f"'model.mtp_layers.' or 'mtp.' were found. "
            f"First {len(sample)} keys: {sample}"
        )

    def _should_extract(self, key: str, fmt: _Format) -> bool:
        """Return True if this key should be included in the converted checkpoint."""
        if fmt == "native_hf":
            return (
                key in _NATIVE_HF_EMBED_KEYS
                or key in _NATIVE_HF_LM_HEAD_KEYS
                or key.startswith("model.mtp_layers.0.")
            )
        return (
            key in _TENCENTBAC_EMBED_KEYS
            or key in _TENCENTBAC_LM_HEAD_KEYS
            or key.startswith("mtp.")
        )

    def _remap_key(self, key: str, fmt: _Format) -> str:
        """Map a source checkpoint key to the FastMTPSpeculator parameter name."""
        if fmt == "native_hf":
            # Strip "model." prefix — parameters live at e.g. "mtp_layers.0.X".
            return key.removeprefix("model.")

        # tencentbac: embed/lm_head may appear with or without the "model." prefix;
        # remaining MTP keys follow the same rules as _fix_state_dict_key_on_load.
        if key in _TENCENTBAC_EMBED_KEYS:
            return "embed_tokens.weight"
        if key in _TENCENTBAC_LM_HEAD_KEYS:
            return "lm_head.weight"
        remapped, _ = FastMTPSpeculator._fix_state_dict_key_on_load(key)  # noqa: SLF001
        return remapped

    def _extract_weights(
        self, checkpoint_dir: Path, all_keys: list[str], fmt: _Format
    ) -> dict[str, torch.Tensor]:
        """Stream only the needed keys from the checkpoint (shard-aware)."""
        needed = {k for k in all_keys if self._should_extract(k, fmt)}

        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            return self._extract_from_shards(checkpoint_dir, index_path, needed, fmt)

        single = checkpoint_dir / "model.safetensors"
        if single.exists():
            weights: dict[str, torch.Tensor] = {}
            with safe_open(str(single), framework="pt") as f:
                available = set(f.keys())  # noqa: SIM118
                for key in needed & available:
                    weights[self._remap_key(key, fmt)] = f.get_tensor(key)
            return weights

        pytorch = checkpoint_dir / "pytorch_model.bin"
        if pytorch.exists():
            state_dict = torch.load(str(pytorch), map_location="cpu", weights_only=True)
            return {
                self._remap_key(k, fmt): v for k, v in state_dict.items() if k in needed
            }

        raise FileNotFoundError(f"No checkpoint weights found at {checkpoint_dir}")

    def _extract_from_shards(
        self,
        checkpoint_dir: Path,
        index_path: Path,
        needed_keys: set[str],
        fmt: _Format,
    ) -> dict[str, torch.Tensor]:
        """Open only the shards that contain the requested keys."""
        with index_path.open() as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        shard_to_keys: dict[str, list[str]] = {}
        for key in needed_keys:
            shard_filename = weight_map.get(key)
            if shard_filename is not None:
                shard_to_keys.setdefault(shard_filename, []).append(key)

        weights: dict[str, torch.Tensor] = {}
        for shard_filename, keys in shard_to_keys.items():
            shard_path = checkpoint_dir / shard_filename
            logger.debug(f"Reading {len(keys)} key(s) from shard {shard_filename}")
            with safe_open(str(shard_path), framework="pt") as f:
                for key in keys:
                    weights[self._remap_key(key, fmt)] = f.get_tensor(key)

        return weights

    def _build_config(
        self, source_config: dict, base_model: str, num_speculative_steps: int
    ) -> FastMTPConfig:
        """Build a FastMTPConfig from the source checkpoint config dict."""
        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        verifier_config = VerifierConfig(
            name_or_path=base_model,
            architectures=verifier_config_dict.get("architectures", []),
        )

        speculators_config = SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=num_speculative_steps)
            ],
            default_proposal_method="greedy",
            verifier=verifier_config,
        )

        # FastMTPConfig's Pydantic validator converts the dict to a PretrainedConfig.
        return FastMTPConfig(
            transformer_layer_config=source_config,  # type: ignore[arg-type]
            speculators_config=speculators_config,
        )

    def _save(
        self,
        config: FastMTPConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = FastMTPSpeculator(config)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected keys during conversion — checkpoint weights do not match "
                f"the model architecture.  This usually means the layer type resolved "
                f"from the config does not match the checkpoint (e.g. linear_attn vs "
                f"self_attn in Qwen3-Next).  Unexpected keys: {unexpected}"
            )
        if missing:
            logger.debug(f"Keys not in extracted weights (filled by init): {missing}")

        # Preserve the source dtype (usually bfloat16 for large models).
        float_dtypes = {t.dtype for t in weights.values() if t.is_floating_point()}
        if float_dtypes:
            model.to(dtype=next(iter(float_dtypes)))  # type: ignore[call-arg]

        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        """Load the converted checkpoint and verify no unexpected keys are missing."""
        logger.info("Validating converted FastMTP checkpoint...")
        try:
            FastMTPSpeculator.from_pretrained(str(output_path))
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
