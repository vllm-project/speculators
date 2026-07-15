"""DFlash checkpoint converter.

Converts an external DFlash checkpoint (e.g. ``z-lab/*-DFlash``) to a Speculators
checkpoint that loads with ``DFlashDraftModel.from_pretrained(path)``.

The draft transformer body (``layers.*``, ``fc``, ``hidden_norm``, ``norm``) already
matches ``DFlashDraftModel`` so weights are copied as-is. The external checkpoint
borrows the verifier's embedding and LM head at runtime, so ``embed_tokens`` /
``lm_head`` / ``verifier_lm_head`` / ``verifier_norm`` are loaded from the verifier
before saving.
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["DFlashConverter"]

# config.json keys that are not part of the draft transformer (Qwen3) config
_NON_TRANSFORMER_KEYS = frozenset(
    {"architectures", "auto_map", "block_size", "dflash_config", "num_target_layers"}
)

# state dict keys that are filled from the verifier (not the source checkpoint), so
# their absence from the source weights is expected, not a conversion error
_VERIFIER_FILLED_KEYS = frozenset(
    {
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    }
)


class DFlashConverter:
    """Convert an external DFlash checkpoint to speculators format.

    Copies the draft transformer body as-is and fills the embedding, LM head, and
    verifier norm from the verifier model so the saved checkpoint is self-contained.
    """

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        aux_hidden_state_layer_ids: list[int] | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
    ) -> None:
        logger.info(f"Converting DFlash checkpoint: {input_path}")

        local_checkpoint_path = ensure_checkpoint_is_local(
            input_path,
            cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        source_config = load_checkpoint_config(local_checkpoint_path)
        weights = load_checkpoint_weights(local_checkpoint_path)
        logger.info(f"Loaded {len(weights)} weights")

        config = self._build_config(
            source_config, base_model, aux_hidden_state_layer_ids
        )
        saved_path = self._save(config, weights, output_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate(saved_path)

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        aux_hidden_state_layer_ids: list[int] | None,
    ) -> DFlashSpeculatorConfig:
        dflash = source_config.get("dflash_config", {})
        # block_size lives at the top level in older checkpoints (Qwen3, LLaMA)
        # and inside dflash_config in newer ones (Qwen3.5+)
        block_size = source_config.get("block_size") or dflash.get("block_size")
        if block_size is None:
            raise ValueError(
                "Checkpoint config has no `block_size` (checked both top-level "
                "and `dflash_config`)"
            )
        transformer_config = {
            k: v for k, v in source_config.items() if k not in _NON_TRANSFORMER_KEYS
        }

        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        source_hidden = transformer_config.get("hidden_size")
        target_hidden = verifier_config_dict.get("hidden_size")
        if source_hidden and target_hidden and source_hidden != target_hidden:
            raise ValueError(
                f"Architecture mismatch: source DFlash checkpoint has "
                f"hidden_size={source_hidden} but base_model '{base_model}' has "
                f"hidden_size={target_hidden}. Dimensions must match."
            )

        if aux_hidden_state_layer_ids is None:
            target_layer_ids = dflash.get("target_layer_ids")
            if target_layer_ids is None:
                raise ValueError(
                    "Checkpoint config has no `dflash_config.target_layer_ids`; "
                    "pass `aux_hidden_state_layer_ids` explicitly."
                )
            # z-lab reads hidden_states[layer_id + 1] (index 0 is the embedding
            # output) while speculators uses the layer id directly.
            # Source: z-lab utils.extract_context_feature.
            aux_hidden_state_layer_ids = [i + 1 for i in target_layer_ids]

        speculators_config = SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    speculative_tokens=block_size - 1,
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=base_model,
                architectures=verifier_config_dict.get("architectures", []),
            ),
        )

        return DFlashSpeculatorConfig(
            transformer_layer_config=transformer_config,  # type: ignore[arg-type]
            draft_vocab_size=transformer_config["vocab_size"],
            block_size=block_size,
            aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
            mask_token_id=dflash.get("mask_token_id"),
            speculators_config=speculators_config,
        )

    def _save(
        self,
        config: DFlashSpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = DFlashDraftModel(config=config)

        body = {k: v for k, v in weights.items() if k not in ("t2d", "d2t")}
        missing, unexpected = model.load_state_dict(body, strict=False)
        if unexpected:
            raise ValueError(
                "Unexpected keys in checkpoint -- the structure does not match "
                f"DFlashDraftModel. Unexpected keys: {unexpected}"
            )
        critical_missing = [k for k in missing if k not in _VERIFIER_FILLED_KEYS]
        if critical_missing:
            raise ValueError(f"Draft weights missing after load: {critical_missing}")
        logger.debug(f"Keys loaded from verifier at save time: {missing}")

        # embed_tokens / lm_head / verifier_lm_head / verifier_norm come from the
        # verifier; without this they would be saved as NaN.
        model.load_verifier_weights()

        model.to(dtype=next(iter(body.values())).dtype)  # type: ignore[call-arg]
        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        logger.info("Validating converted DFlash checkpoint...")
        try:
            model = DFlashDraftModel.from_pretrained(str(output_path))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
        for name in ("fc.weight", "lm_head.weight", "embed_tokens.weight"):
            if torch.isnan(model.state_dict()[name]).any():
                raise ValueError(f"Converted checkpoint has NaN in {name}")
        logger.success("Validation succeeded")
