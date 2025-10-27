"""
Eagle-3 checkpoint converter with loguru logging.
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import LlamaConfig, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
    find_vocab_size,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.eagle3 import Eagle3Speculator, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


class Eagle3Converter:
    """
    Converter for Eagle3 checkpoints to speculators format.

    Handles weight remapping, embeddings replacement, and vLLM compatibility fixes.
    Produces production-ready models with standardized speculators_config metadata.
    """

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        norm_before_residual: bool = False,
        eagle_aux_hidden_state_layer_ids: list[int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting Eagle-3 checkpoint: {input_path}")

        local_checkpoint_path = ensure_checkpoint_is_local(input_path, cache_dir)

        eagle_config = load_checkpoint_config(local_checkpoint_path)
        weights = load_checkpoint_weights(local_checkpoint_path)
        logger.info(f"Loaded {len(weights)} weights")

        reduce_vocab_size = False
        # Get target_vocab_size from t2d tensor shape if available
        if "t2d" in weights:
            eagle_config["target_vocab_size"] = weights["t2d"].shape[0]
            logger.debug(
                f"Using target_vocab_size from t2d tensor: "
                f"{eagle_config['target_vocab_size']}"
            )
            reduce_vocab_size = True
        else:
            # fall back to target model config - search for vocab_size at any level
            target_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
            vocab_size = find_vocab_size(target_config_dict)
            if vocab_size is None:
                raise ValueError(
                    "Could not determine vocab_size from target model config."
                )
            eagle_config["target_vocab_size"] = vocab_size
            logger.debug(
                f"Using target_vocab_size from config: "
                f"{eagle_config['target_vocab_size']}"
            )

        config = self._build_eagle3_speculator_config(
            eagle_config,
            base_model,
            norm_before_residual,
            eagle_aux_hidden_state_layer_ids,
        )

        has_drafter_embedding = "embed_tokens.weight" in weights

        saved_path = self._save_converted_checkpoint(
            config,
            weights,
            output_path,
            reduce_vocab_size,
            has_drafter_embedding,
        )
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate_converted_checkpoint(saved_path, base_model)

    def _create_verifier_config(self, base_model: str) -> VerifierConfig:
        config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        return VerifierConfig(
            name_or_path=base_model,
            architectures=config_dict.get("architectures", ["LlamaForCausalLM"]),
        )

    def _build_eagle3_speculator_config(
        self,
        eagle_config: dict,
        base_model: str,
        norm_before_residual: bool = False,
        eagle_aux_hidden_state_layer_ids: list[int] | None = None,
    ) -> Eagle3SpeculatorConfig:
        transformer_config = self._create_transformer_config_from_eagle(
            eagle_config, base_model
        )
        verifier_config = self._create_verifier_config(base_model)

        proposal_config = GreedyTokenProposalConfig(
            proposal_type="greedy",
            speculative_tokens=3,
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
            norm_before_residual=norm_before_residual,
            target_hidden_size=eagle_config.get("target_hidden_size"),
            eagle_aux_hidden_state_layer_ids=eagle_aux_hidden_state_layer_ids,
        )

    def _create_transformer_config_from_eagle(
        self, eagle_config: dict, base_model: str
    ) -> LlamaConfig:
        # Load target model config for vLLM compatibility
        try:
            target_config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load config for base model {base_model}: {e}"
            ) from e

        return LlamaConfig(
            vocab_size=eagle_config.get("target_vocab_size", 128000),
            hidden_size=eagle_config.get("hidden_size", 4096),
            intermediate_size=eagle_config.get("intermediate_size", 11008),
            num_hidden_layers=eagle_config.get("num_hidden_layers", 1),
            num_attention_heads=eagle_config.get("num_attention_heads", 32),
            num_key_value_heads=eagle_config.get("num_key_value_heads", 8),
            hidden_act=eagle_config.get("hidden_act", "silu"),
            # Ensure max_position_embeddings match between Eagle3 and target configs
            max_position_embeddings=max(
                eagle_config.get("max_position_embeddings", 4096),
                target_config_dict.get("max_position_embeddings", 4096),
            ),
            initializer_range=eagle_config.get("initializer_range", 0.02),
            rms_norm_eps=eagle_config.get("rms_norm_eps", 1e-6),
            use_cache=True,
            attention_bias=eagle_config.get("attention_bias", False),
            rope_theta=eagle_config.get("rope_theta", 10000.0),
            mlp_bias=eagle_config.get("mlp_bias", False),
            tie_word_embeddings=False,
            torch_dtype=eagle_config.get("torch_dtype"),
            head_dim=eagle_config.get("head_dim"),
        )

    def _save_converted_checkpoint(
        self,
        config: Eagle3SpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_dir: str | Path,
        reduce_vocab_size: bool,
        has_drafter_embedding: bool,
    ) -> Path:
        model = Eagle3Speculator(
            config=config,
            verifier=None,
            verifier_attachment_mode="detached",
            reduce_vocab_size=reduce_vocab_size,
            has_drafter_embedding=has_drafter_embedding,
        )
        model.load_state_dict(weights, strict=False)  # type: ignore[attr-defined]
        weights_dtype = getattr(config.transformer_layer_config, "torch_dtype", None)
        # .to() wont convert d2t/t2d buffers as they are not fp tensors
        model.to(dtype=weights_dtype)  # type: ignore[call-arg]
        model.save_pretrained(str(output_dir))  # type: ignore[attr-defined]
        return Path(output_dir)

    def _validate_converted_checkpoint(
        self, checkpoint_path: Path, base_model: str
    ) -> None:
        logger.info("Validating converted Eagle-3 checkpoint...")
        try:
            Eagle3Speculator.from_pretrained(
                checkpoint_path,
                verifier=base_model,
                verifier_attachment_mode="detached",
            )
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Validation failed: {e}")
            raise
