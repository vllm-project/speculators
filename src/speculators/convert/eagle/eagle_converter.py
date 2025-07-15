"""
Eagle checkpoint converter with loguru logging.
"""

import os

from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from transformers import LlamaConfig


from speculators.convert.eagle.utils import (
    detect_fusion_bias_and_layernorms,
)
from speculators.convert.eagle.base import SpeculatorConverter
from speculators.models.eagle import EagleSpeculator, EagleSpeculatorConfig
from transformers import LlamaConfig, PreTrainedModel

@SpeculatorConverter.register("eagle")
class EagleConverter(SpeculatorConverter):
    """
    Converter for Eagle/HASS checkpoints to speculators format.

    This converter handles the transformation of Eagle-style checkpoints
    (including HASS variants) into the standardized speculators format.
    It supports automatic feature detection, weight remapping, and
    optional validation.

    :Example:

        >>> converter = EagleConverter()
        >>> converter.convert(
        ...     "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        ...     "./output",
        ...     "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ... )
    """

    EAGLE_TO_SPECULATORS_LAYERNORM_MAPPINGS = {
        "embed_layernorm.weight": "embedding_layernorm.weight",
        "lm_head_layernorm.weight": "pre_lm_head_layernorm.weight",
    }

    def __init__(
        self,
        model: Union[Path, PreTrainedModel, torch.nn.Module, str],
        verifier: Union[str, os.PathLike, PreTrainedModel],
        output_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):

        super().__init__(
            model=model,
            verifier=verifier,
            output_path=output_path,
            model_type="eagle",
            cache_dir=cache_dir,
        )

    def convert(
        self,
        fusion_bias: bool = False,
        layernorms: bool = False,
        validate: bool = True,
    ) -> None:
        """
        Convert an Eagle checkpoint to speculators format.

        This method orchestrates the complete conversion process:

        1. Ensures the checkpoint is available locally
        2. Loads the original config and weights
        3. Auto-detects features if not explicitly specified (layernorms, fusion bias)
        4. Builds the speculators configuration
        5. Processes and remaps the weights
        6. Saves the converted checkpoint
        7. Optionally validates the result by running a forward pass

        :param fusion_bias: Enable fusion bias (auto-detected if not specified)
        :param layernorms: Enable extra layernorms (auto-detected if not specified)
        :param validate: Whether to validate the converted checkpoint

        :Example:

            >>> # Convert standard Eagle checkpoint
            >>> converter = EagleConverter()
            >>> converter.convert(
            ...     "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
            ...     "./eagle-converted",
            ...     "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ...     validate=True
            ... )

            >>> # Convert HASS checkpoint with layernorms
            >>> converter.convert(
            ...     "nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT",
            ...     "./hass-converted",
            ...     "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ...     layernorms=True
            ... )
        """

        detected_fusion_bias, detected_layernorms = detect_fusion_bias_and_layernorms(
            self.weights
        )
        fusion_bias = fusion_bias or detected_fusion_bias
        layernorms = layernorms or detected_layernorms

        speculator_config = self._build_eagle_speculator_config(
            self.config, self.verifier, fusion_bias, layernorms
        )

        processed_weights = self._process_checkpoint_weights(self.weights, layernorms)

        # Save the converted checkpoint using the model's save_pretrained
        saved_path = self._save_converted_checkpoint(
            config=speculator_config, weights=processed_weights, output_dir=self.output_path
        )

        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate_converted_checkpoint(saved_path, verifier_model=self.verifier)

    def _create_transformer_config_from_eagle(self, eagle_config: dict) -> LlamaConfig:
        """
        Create a transformer config for the Eagle model's single decoder layer.

        :param eagle_config: Original Eagle checkpoint config
        :return: LlamaConfig for the transformer layer
        """
        return LlamaConfig(
            vocab_size=eagle_config.get("vocab_size", 32000),
            hidden_size=eagle_config.get("hidden_size", 4096),
            intermediate_size=eagle_config.get("intermediate_size", 11008),
            num_hidden_layers=1,  # Eagle always uses a single decoder layer
            num_attention_heads=eagle_config.get("num_attention_heads", 32),
            num_key_value_heads=eagle_config.get("num_key_value_heads"),
            hidden_act=eagle_config.get("hidden_act", "silu"),
            max_position_embeddings=eagle_config.get("max_position_embeddings", 4096),
            initializer_range=eagle_config.get("initializer_range", 0.02),
            rms_norm_eps=eagle_config.get("rms_norm_eps", 1e-6),
            use_cache=eagle_config.get("use_cache", True),
            pad_token_id=eagle_config.get("pad_token_id"),
            bos_token_id=eagle_config.get("bos_token_id", 1),
            eos_token_id=eagle_config.get("eos_token_id", 2),
            tie_word_embeddings=False,  # Eagle uses separate embed_tokens from verifier
            rope_theta=eagle_config.get("rope_theta", 10000.0),
            rope_scaling=eagle_config.get("rope_scaling"),
            attention_bias=eagle_config.get("attention_bias", False),
            attention_dropout=eagle_config.get("attention_dropout", 0.0),
            mlp_bias=eagle_config.get("mlp_bias", False),
        )

    def _build_eagle_speculator_config(
        self,
        eagle_config: dict,
        base_model: str,
        fusion_bias: bool,
        layernorms: bool,
    ) -> EagleSpeculatorConfig:
        """
        Build a complete EagleSpeculatorConfig from Eagle checkpoint config.

        :param eagle_config: Original checkpoint config dictionary
        :param base_model: Base model name for the verifier
        :param fusion_bias: Whether to enable fusion bias
        :param layernorms: Whether to enable extra layernorms
        :return: Complete Eagle speculator configuration
        """
        logger.debug(
            f"Building config with fusion_bias={fusion_bias}, layernorms={layernorms}"
        )

        transformer_config = self._create_transformer_config_from_eagle(eagle_config)

        speculators_config = self._build_speculator_config(
            checkpoint_config=eagle_config,
            base_model=base_model,
        )

        return EagleSpeculatorConfig(
            transformer_layer_config=transformer_config,
            speculators_config=speculators_config,
            layernorms=layernorms,
            fusion_bias=fusion_bias,
        )

    def _should_skip_weight(self, weight_name: str, has_layernorms: bool) -> bool:
        """
        Determine if a weight should be skipped during conversion.

        :param weight_name: Original weight name
        :param has_layernorms: Whether layernorms are enabled
        :return: True if the weight should be excluded from the output
        """
        # Skip embed_tokens - Eagle gets these from the verifier model
        if weight_name == "embed_tokens.weight":
            logger.debug("Skipping embed_tokens.weight (tied to lm_head)")
            return True

        # Skip hidden_layernorm when layernorms are disabled
        return weight_name == "hidden_layernorm.weight" and not has_layernorms

    def _remap_weight_name(self, weight_name: str, has_layernorms: bool) -> str:
        """
        Remap an Eagle weight name to speculators format.

        :param weight_name: Original weight name
        :param has_layernorms: Whether layernorms are enabled
        :return: Remapped weight name
        """
        # hidden_layernorm maps to the decoder's input_layernorm when layernorms enabled
        if weight_name == "hidden_layernorm.weight" and has_layernorms:
            return "transformer.input_layernorm.weight"

        if (
            has_layernorms
            and weight_name in self.EAGLE_TO_SPECULATORS_LAYERNORM_MAPPINGS
        ):
            return self.EAGLE_TO_SPECULATORS_LAYERNORM_MAPPINGS[weight_name]

        if weight_name.startswith("fc."):
            return weight_name.replace("fc.", "fusion_fc.")

        if weight_name.startswith("layers.0."):
            return weight_name.replace("layers.0.", "transformer.")

        return weight_name

    def _process_checkpoint_weights(
        self,
        weights: dict[str, torch.Tensor],
        has_layernorms: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Process and remap all weights from Eagle to speculators format.

        :param weights: Original checkpoint weights
        :param has_layernorms: Whether layernorms are enabled
        :return: Processed weights with remapped names
        """
        logger.debug(f"Processing {len(weights)} weights")

        processed_weights = {}
        skipped_weights = []
        remapped_weights = []

        for original_name, tensor in weights.items():
            if self._should_skip_weight(original_name, has_layernorms):
                skipped_weights.append(original_name)
                continue

            new_name = self._remap_weight_name(original_name, has_layernorms)
            processed_weights[new_name] = tensor

            if new_name != original_name:
                remapped_weights.append(f"{original_name} -> {new_name}")

        if skipped_weights:
            logger.debug(f"Skipped weights: {skipped_weights}")
        if remapped_weights:
            logger.debug(f"Remapped weights: {remapped_weights}")

        return processed_weights
