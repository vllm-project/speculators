"""
Eagle-3 checkpoint converter with loguru logging.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, LlamaConfig, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
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
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        base_model: str,
        validate: bool = True,
        norm_before_residual: bool = False,
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
            eagle_config,
            base_model,
            norm_before_residual,
        )

        # Process weights and ensure embeddings are properly handled
        processed_weights = self._process_checkpoint_weights(weights, base_model)

        saved_path = self._save_converted_checkpoint(
            config, processed_weights, output_path
        )
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate_converted_checkpoint(saved_path, base_model)

    def _process_checkpoint_weights(
        self, weights: dict[str, torch.Tensor], base_model: str
    ) -> dict[str, torch.Tensor]:
        """
        Process and validate Eagle3 checkpoint weights.

        Eagle3 models need embeddings that match the verifier model for good acceptance.
        We ALWAYS replace embeddings with verifier embeddings for compatibility.

        :param weights: Original checkpoint weights
        :param base_model: Base model name to load verifier embeddings from
        :return: Processed weights with verifier embeddings
        """
        logger.debug(f"Processing {len(weights)} Eagle3 weights")

        # Remap weight names: midlayer.* -> layers.0.*
        processed_weights = {}
        for original_name, tensor in weights.items():
            # Remap midlayer.* -> layers.0.*
            if original_name.startswith("midlayer."):
                new_name = original_name.replace("midlayer.", "layers.0.")
                processed_weights[new_name] = tensor
                logger.debug(f"Remapped: {original_name} -> {new_name}")
            # Keep layers.0.* as is (already correct)
            elif original_name.startswith("layers.0."):
                processed_weights[original_name] = tensor
            else:
                processed_weights[original_name] = tensor

        # Only add verifier embeddings if not present in eagle model
        if "embed_tokens.weight" not in processed_weights:
            logger.info("Eagle model missing embeddings - adding verifier embeddings")
            return self._add_verifier_embeddings(processed_weights, base_model)
        else:
            logger.info("Eagle model already has embeddings - keeping originals")
            return processed_weights

    def _add_verifier_embeddings(
        self, weights: dict[str, torch.Tensor], base_model: str
    ) -> dict[str, torch.Tensor]:
        """
        Add embeddings from the verifier model to the checkpoint.

        :param weights: Current checkpoint weights
        :param base_model: Base model to load embeddings from
        :return: Updated weights with verifier embeddings
        """
        logger.info(f"Loading embeddings from verifier model: {base_model}")

        try:
            # Load verifier model to get embeddings
            verifier = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float32
            )

            # Extract embeddings from verifier
            if hasattr(verifier, "model") and hasattr(verifier.model, "embed_tokens"):
                embed_tokens = verifier.model.embed_tokens.weight.data.clone()
            elif hasattr(verifier, "embed_tokens"):
                embed_tokens = verifier.embed_tokens.weight.data.clone()
            else:
                raise RuntimeError(
                    f"Could not find embed_tokens in verifier model {base_model}"
                )

            logger.info(f"Loaded embeddings with shape: {embed_tokens.shape}")
            weights["embed_tokens.weight"] = embed_tokens

            # Clean up verifier model to save memory
            del verifier
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to load embeddings from verifier: {e}")
            raise RuntimeError(
                f"Could not load embeddings from verifier model {base_model}. "
                "This is required for Eagle3 models without trained embeddings."
            ) from e

        return weights

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
    ) -> Eagle3SpeculatorConfig:
        transformer_config = self._create_transformer_config_from_eagle(
            eagle_config, base_model
        )
        verifier_config = self._create_verifier_config(base_model)

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
            norm_before_residual=norm_before_residual,
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
            num_hidden_layers=1,
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
