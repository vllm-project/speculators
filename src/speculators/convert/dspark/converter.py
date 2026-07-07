"""DSpark checkpoint converter.

Converts an external DSpark checkpoint to a Speculators checkpoint that loads with
``DSparkDraftModel.from_pretrained(path)``.

DSpark is DFlash plus a sequential (Markov) logit-bias head and a per-position
confidence head (arXiv:2607.05147). The draft transformer body (``layers.*``,
``fc``, ``hidden_norm``, ``norm``), the Markov head (``markov_head.*``), and the
confidence head (``confidence_head.*``) already match ``DSparkDraftModel``, so the
weights are copied as-is. As with DFlash, the draft borrows the verifier's
embedding and LM head at runtime, so ``embed_tokens`` / ``lm_head`` /
``verifier_lm_head`` / ``verifier_norm`` are loaded from the verifier before saving.

Unlike DFlash checkpoints, the DSpark config keeps its algorithm fields flat at the
top level (there is no nested ``dflash_config`` block).
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
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["DSparkConverter"]

# config.json keys that are not part of the draft transformer (Qwen3) config.
# DSpark stores its algorithm fields flat at the top level (no ``dflash_config``).
_NON_TRANSFORMER_KEYS = frozenset(
    {
        "architectures",
        "auto_map",
        "block_size",
        "mask_token_id",
        "target_layer_ids",
        "num_anchors",
        "num_target_layers",
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
    }
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


class DSparkConverter:
    """Convert an external DSpark checkpoint to speculators format.

    Copies the draft transformer body, Markov head, and confidence head as-is and
    fills the embedding, LM head, and verifier norm from the verifier model so the
    saved checkpoint is self-contained.
    """

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        aux_hidden_state_layer_ids: list[int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting DSpark checkpoint: {input_path}")

        local_checkpoint_path = ensure_checkpoint_is_local(input_path, cache_dir)
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
    ) -> DSparkSpeculatorConfig:
        block_size = source_config.get("block_size")
        if block_size is None:
            raise ValueError("Checkpoint config has no top-level `block_size`")
        transformer_config = {
            k: v for k, v in source_config.items() if k not in _NON_TRANSFORMER_KEYS
        }

        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        source_hidden = transformer_config.get("hidden_size")
        target_hidden = verifier_config_dict.get("hidden_size")
        if source_hidden and target_hidden and source_hidden != target_hidden:
            raise ValueError(
                f"Architecture mismatch: source DSpark checkpoint has "
                f"hidden_size={source_hidden} but base_model '{base_model}' has "
                f"hidden_size={target_hidden}. Dimensions must match."
            )

        if aux_hidden_state_layer_ids is None:
            target_layer_ids = source_config.get("target_layer_ids")
            if target_layer_ids is None:
                raise ValueError(
                    "Checkpoint config has no top-level `target_layer_ids`; "
                    "pass `aux_hidden_state_layer_ids` explicitly."
                )
            # The draft reads hidden_states[layer_id + 1] (index 0 is the embedding
            # output), with layer_id == -1 selecting the embedding output directly,
            # while speculators uses the hidden_states index.
            aux_hidden_state_layer_ids = [
                0 if i == -1 else i + 1 for i in target_layer_ids
            ]

        speculators_config = SpeculatorsConfig(
            algorithm="dspark",
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

        return DSparkSpeculatorConfig(
            transformer_layer_config=transformer_config,  # type: ignore[arg-type]
            draft_vocab_size=transformer_config["vocab_size"],
            block_size=block_size,
            aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
            mask_token_id=source_config.get("mask_token_id"),
            markov_rank=source_config.get("markov_rank", 256),
            markov_head_type=source_config.get("markov_head_type", "vanilla"),
            enable_confidence_head=source_config.get("enable_confidence_head", True),
            confidence_head_with_markov=source_config.get(
                "confidence_head_with_markov", True
            ),
            speculators_config=speculators_config,
        )

    def _save(
        self,
        config: DSparkSpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = DSparkDraftModel(config=config)

        body = {k: v for k, v in weights.items() if k not in ("t2d", "d2t")}
        missing, unexpected = model.load_state_dict(body, strict=False)
        if unexpected:
            raise ValueError(
                "Unexpected keys in checkpoint -- the structure does not match "
                f"DSparkDraftModel. Unexpected keys: {unexpected}"
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
        logger.info("Validating converted DSpark checkpoint...")
        try:
            model = DSparkDraftModel.from_pretrained(str(output_path))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
        state_dict = model.state_dict()
        names = ["fc.weight", "lm_head.weight", "embed_tokens.weight"]
        if getattr(model, "markov_head", None) is not None:
            names.append("markov_head.markov_w2.weight")
        if getattr(model, "confidence_head", None) is not None:
            names.append("confidence_head.proj.weight")
        for name in names:
            if torch.isnan(state_dict[name]).any():
                raise ValueError(f"Converted checkpoint has NaN in {name}")
        logger.success("Validation succeeded")
