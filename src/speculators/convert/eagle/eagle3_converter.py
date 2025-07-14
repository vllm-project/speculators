"""
Eagle-3 checkpoint converter with loguru logging.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    detect_fusion_bias_and_layernorms,
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.eagle3 import Eagle3Speculator, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


class Eagle3Converter:
    """
    Converter for Eagle-3 checkpoints to speculators format.
    Supports automatic feature detection, weight remapping, and optional validation.
    """

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        base_model: str,
        validate: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        logger.info(f"Converting Eagle-3 checkpoint: {input_path}")

        local_checkpoint_path = ensure_checkpoint_is_local(input_path, cache_dir)

        eagle_config = load_checkpoint_config(local_checkpoint_path)
        weights = load_checkpoint_weights(local_checkpoint_path)
        logger.info(f"Loaded {len(weights)} weights")

        # Patch: ensure target_vocab_size matches t2d tensor shape
        eagle_config["target_vocab_size"] = weights["t2d"].shape[0]

        config = self._build_eagle3_speculator_config(
            eagle_config, base_model,
        )

        saved_path = self._save_converted_checkpoint(config, weights, output_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate_converted_checkpoint(saved_path, base_model)


    def _build_eagle3_speculator_config(
        self,
        eagle_config: dict,
        base_model: str,
    ) -> Eagle3SpeculatorConfig:
        transformer_config = self._create_transformer_config_from_eagle(eagle_config)
        verifier_config = self._create_verifier_config_from_eagle(eagle_config, 
                                                                  base_model)

        proposal_config = GreedyTokenProposalConfig(
            proposal_type="greedy",
            speculative_tokens=5,
        )

        speculators_config = SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[proposal_config],
            default_proposal_method="greedy",
            verifier=verifier_config,
        )

        return Eagle3SpeculatorConfig(
            transformer_layer_config=transformer_config,
            speculators_config=speculators_config,
            draft_vocab_size=eagle_config.get("draft_vocab_size", 32000),
            norm_before_residual=eagle_config.get("norm_before_residual", False),
        )

    def _create_transformer_config_from_eagle(self, eagle_config: dict) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=eagle_config.get("target_vocab_size", 128000),
            hidden_size=eagle_config.get("hidden_size", 4096),
            intermediate_size=eagle_config.get("intermediate_size", 11008),
            num_hidden_layers=1,
            num_attention_heads=eagle_config.get("num_attention_heads", 32),
            num_key_value_heads=eagle_config.get("num_key_value_heads", 8),
            hidden_act=eagle_config.get("hidden_act", "silu"),
            max_position_embeddings=eagle_config.get("max_position_embeddings", 4096),
            initializer_range=eagle_config.get("initializer_range", 0.02),
            rms_norm_eps=eagle_config.get("rms_norm_eps", 1e-6),
            use_cache=True,
            attention_bias=eagle_config.get("attention_bias", False),
            rope_theta=eagle_config.get("rope_theta", 10000.0),
            mlp_bias=eagle_config.get("mlp_bias", False),
            tie_word_embeddings=False,
        )

    def _create_verifier_config_from_eagle(
            self, 
            eagle_config: dict, 
            base_model: str
    ) -> VerifierConfig:
        eos_token_id = eagle_config.get("eos_token_id", 2)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        return VerifierConfig(
            name_or_path=base_model,
            architectures=eagle_config.get("architectures", ["LlamaForCausalLM"]),
        )

    def _save_converted_checkpoint(
        self,
        config: Eagle3SpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_dir: Union[str, Path],
    ) -> Path:
        model = Eagle3Speculator(
            config=config,
            verifier=None,
            verifier_attachment_mode="detached",
        )
        model.load_state_dict(weights, strict=False)  # type: ignore[attr-defined]
        model.save_pretrained(str(output_dir))  # type: ignore[attr-defined]
        return Path(output_dir)

    def _validate_converted_checkpoint(
            self, 
            checkpoint_path: Path, 
            base_model: str
    ) -> None:
        logger.info("Validating converted Eagle-3 checkpoint...")
        try:
            Eagle3Speculator.from_pretrained(
                checkpoint_path,
                verifier=base_model,
                verifier_attachment_mode="detached",
            )
            logger.success("Validation succeeded")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
