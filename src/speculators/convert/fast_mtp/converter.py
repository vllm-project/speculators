"""FastMTP checkpoint converter for Qwen3-Next.

Extracts the MTP layer (plus embed_tokens and lm_head) from a Qwen3-Next
checkpoint and saves a self-contained Speculators checkpoint that loads with
``FastMTPSpeculator.from_pretrained(path)`` without the full base model.

Qwen3-Next stores its MTP layer under the ``mtp.*`` key prefix, alongside the
main model weights.  Only that subtree — plus ``embed_tokens`` and ``lm_head``
— is read from the (potentially sharded) safetensors file; the rest of the
model is never loaded into memory.
"""

import json
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
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["FastMTPConverter"]

_MTP_PREFIX = "mtp."
_EMBED_KEY = "model.embed_tokens.weight"
_LM_HEAD_KEY = "model.lm_head.weight"


class FastMTPConverter:
    """Extract the FastMTP head from a Qwen3-Next checkpoint.

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
        """Convert a Qwen3-Next checkpoint to the Speculators FastMTP format.

        :param input_path: Local path or Hugging Face model ID for the source.
        :param output_path: Directory where the converted checkpoint will be written.
        :param base_model: Verifier model ID or path; stored in VerifierConfig.
        :param num_speculative_steps: Number of MTP steps to configure in the output.
        :param validate: Load the output checkpoint after saving to verify correctness.
        :param cache_dir: Optional cache directory for Hub downloads.
        """
        logger.info(f"Converting FastMTP checkpoint: {input_path}")

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = load_checkpoint_config(local_path)

        all_keys = self._list_checkpoint_keys(local_path)
        self._verify_qwen3_next_format(all_keys)

        weights = self._extract_weights(local_path, all_keys)
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

    def _verify_qwen3_next_format(self, keys: list[str]) -> None:
        """Raise ValueError if the checkpoint lacks Qwen3-Next MTP layer keys."""
        if not any(k.startswith(_MTP_PREFIX) for k in keys):
            raise ValueError(
                f"No MTP layer keys found (expected prefix '{_MTP_PREFIX}'). "
                "This converter supports Qwen3-Next checkpoints only. "
                "Ensure the checkpoint is Qwen/Qwen3-Next-80B-A3B-Instruct or similar."
            )

    def _should_extract(self, key: str) -> bool:
        """Return True if this key belongs to the MTP head or shared embeddings."""
        return key in {_EMBED_KEY, _LM_HEAD_KEY} or key.startswith(_MTP_PREFIX)

    def _remap_key(self, key: str) -> str:
        """Map a source checkpoint key to the FastMTPSpeculator parameter name."""
        _exact = {
            "mtp.fc.weight": "mtp_layers.0.input_proj.weight",
            "mtp.norm.weight": "mtp_layers.0.final_layernorm.weight",
        }
        _prefixes = {
            "model.": "",
            "mtp.pre_fc_norm_hidden.": "mtp_layers.0.hidden_layernorm.",
            "mtp.pre_fc_norm_embedding.": "mtp_layers.0.token_layernorm.",
            "mtp.layers.0.": "mtp_layers.0.",
        }
        if key in _exact:
            return _exact[key]
        for src, dst in _prefixes.items():
            if key.startswith(src):
                return dst + key[len(src) :]
        return key

    def _extract_weights(
        self, checkpoint_dir: Path, all_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        """Stream only the MTP keys from the checkpoint (shard-aware)."""
        needed = {k for k in all_keys if self._should_extract(k)}

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
        """Open only the shards that contain the requested keys."""
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
        self, source_config: dict, base_model: str, num_speculative_steps: int
    ) -> FastMTPConfig:
        """Build a FastMTPConfig from the source checkpoint config dict."""
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
                "Unexpected keys in extracted weights — the checkpoint structure does "
                "not match the model architecture.  This may indicate a wrong "
                "layer_idx for Qwen3-Next (full_attention vs linear_attention). "
                f"Unexpected keys: {unexpected}"
            )
        # Qwen3-Next ties lm_head to embed_tokens (no separate lm_head tensor).
        if "lm_head.weight" not in weights and "embed_tokens.weight" in weights:
            logger.debug("lm_head not in checkpoint; tying to embed_tokens.weight")
            with torch.no_grad():
                model.lm_head.weight.copy_(weights["embed_tokens.weight"])

        missing_non_tied = [k for k in missing if k != "lm_head.weight"]
        if missing_non_tied:
            logger.debug(
                f"Keys not in extracted weights (filled by init): {missing_non_tied}"
            )

        float_dtypes = {t.dtype for t in weights.values() if t.is_floating_point()}
        if float_dtypes:
            model.to(dtype=next(iter(float_dtypes)))  # type: ignore[call-arg]

        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        """Load the converted checkpoint and confirm it deserializes without error."""
        logger.info("Validating converted checkpoint...")
        try:
            FastMTPSpeculator.from_pretrained(str(output_path))
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
