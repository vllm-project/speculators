"""
Eagle/HASS checkpoint converter for Speculators model format.

This module provides the EagleSpeculatorConverter class for transforming Eagle-style
speculative decoding checkpoints (including HASS variants) from research repositories
into the standardized Speculators format. The converter handles automatic feature
detection, weight remapping, configuration translation, and optional validation.

Supported Research Repositories:
    - Eagle v1 and v2: https://github.com/SafeAILab/EAGLE
    - HASS: https://github.com/HArmonizedSS/HASS

Classes:
    EagleSpeculatorConverter: Converter implementation for Eagle/HASS checkpoints

Usage:
::
    from speculators.convert.converters import EagleSpeculatorConverter

    # Convert with automatic feature detection
    converter = EagleSpeculatorConverter(
        model="path/to/eagle_checkpoint",
        config="path/to/config.json",
        verifier="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    converted_model = converter(output_path="./output", validate_device="cuda")
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from torch import Tensor, nn
from transformers import LlamaConfig, PretrainedConfig, PreTrainedModel

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.converters.base import SpeculatorConverter
from speculators.models.eagle import EagleSpeculator, EagleSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils import (
    load_model_checkpoint_config_dict,
    load_model_checkpoint_state_dict,
)

__all__ = ["EagleSpeculatorConverter"]


@SpeculatorConverter.register(["eagle", "eagle2", "hass"])
class EagleSpeculatorConverter(
    SpeculatorConverter[EagleSpeculatorConfig, EagleSpeculator]
):
    """
    Converter for Eagle/HASS research checkpoint format to Speculators format.

    This converter transforms Eagle-style speculative decoding checkpoints into the
    standardized Speculators format, handling weight remapping, configuration
    translation, and feature detection. It supports both the original Eagle
    architecture and its variants including HASS.

    The converter automatically detects model features such as fusion bias and
    layernorms based on the checkpoint structure, with options for manual override.
    """

    WEIGHT_MAPPINGS = {
        "fc.": "fusion_fc.",
        "layers.0.": "transformer.",
    }
    LAYERNORM_MAPPINGS = {
        "embed_layernorm.weight": "embedding_layernorm.weight",
        "hidden_layernorm.weight": "transformer.input_layernorm.weight",
        "lm_head_layernorm.weight": "pre_lm_head_layernorm.weight",
    }

    @classmethod
    def is_supported(
        cls,
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],  # noqa: ARG003
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,  # noqa: ARG003
        fusion_bias: Optional[bool] = None,  # noqa: ARG003
        layernorms: Optional[bool] = None,  # noqa: ARG003
        **kwargs,  # noqa: ARG003
    ) -> bool:
        """
        Check if the provided model checkpoint is supported by this converter.

        Validates that the model follows the Eagle architecture pattern by checking
        for the presence of fusion layer weights and single transformer layer structure.

        :param model: Model checkpoint path or instance to validate
        :param config: Model configuration (unused for Eagle detection)
        :param verifier: Optional verifier model (unused for Eagle detection)
        :param fusion_bias: Optional fusion bias setting (unused for Eagle detection)
        :param layernorms: Optional layernorms setting (unused for Eagle detection)
        :param kwargs: Additional arguments (unused for Eagle detection)
        :return: True if the model follows Eagle architecture pattern
        """
        state_dict = load_model_checkpoint_state_dict(model)
        has_fc = "fc.weight" in state_dict
        has_layers_0 = any(name.startswith("layers.0.") for name in state_dict)
        has_layers_non_0 = any(
            name.startswith("layers.") and not name.startswith("layers.0.")
            for name in state_dict
        )

        return has_fc and has_layers_0 and not has_layers_non_0

    def __init__(
        self,
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        fusion_bias: Optional[bool] = None,
        layernorms: Optional[bool] = None,
    ):
        """
        Initialize the Eagle converter with model, configuration, and feature settings.

        :param model: Model checkpoint path or instance to convert
        :param config: Model configuration path or instance
        :param verifier: Optional verifier model path or instance for speculative
            decoding
        :param fusion_bias: Whether to include fusion bias in conversion. If None,
            automatically detected from checkpoint structure
        :param layernorms: Whether to include extra layernorms in conversion. If None,
            automatically detected from checkpoint structure
        """
        super().__init__(
            model=model,
            config=config,
            verifier=verifier,
        )
        self.fusion_bias = fusion_bias
        self.layernorms = layernorms

    def convert_config_state_dict(
        self,
    ) -> tuple[EagleSpeculatorConfig, dict[str, Tensor]]:
        """
        Convert Eagle/HASS checkpoint configuration and state dict to Speculators
        format.

        Processes the original Eagle checkpoint by detecting features, remapping
        weights, and creating a compatible EagleSpeculatorConfig. Handles automatic
        detection of fusion bias and layernorms based on checkpoint structure.

        :return: Tuple of converted configuration and remapped state dictionary
        """
        logger.info(
            f"Converting Eagle/HASS checkpoint at model: {self.model} and "
            f"config: {self.config} to speculators format..."
        )
        orig_state_dict = load_model_checkpoint_state_dict(self.model)
        orig_config = load_model_checkpoint_config_dict(self.config)
        fusion_bias = (
            self.fusion_bias
            if self.fusion_bias is not None
            else "fc.bias" in orig_state_dict
        )
        layernorms = (
            self.layernorms
            if self.layernorms is not None
            else any(name in orig_state_dict for name in self.LAYERNORM_MAPPINGS)
        )

        converted_config = self._eagle_speculator_config(
            orig_config, fusion_bias, layernorms
        )
        logger.info(
            f"Converted Eagle/HASS config to speculators format: {converted_config}"
        )

        converted_state_dict, missing, extra = self._eagle_speculator_state_dict(
            orig_state_dict, fusion_bias, layernorms
        )
        logger.info(
            "Converted Eagle/HASS state_dict to speculators format: "
            f"{converted_state_dict.keys()}"
        )
        if missing:
            logger.warning(f"Missing keys in converted state_dict: {missing}")
        if extra:
            logger.warning(f"Extra keys in converted state_dict: {extra}")

        return converted_config, converted_state_dict

    def validate(self, model: EagleSpeculator, device: Union[str, torch.device, int]):
        """
        Validate the converted model by running a forward pass with test data.

        Ensures the converted EagleSpeculator model is correctly configured and can
        process inputs without errors. Uses conservative defaults for batch size and
        sequence length to minimize resource requirements.

        :param model: The converted EagleSpeculator model to validate
        :param device: Device for validation (string, torch.device, or device index)
        :raises Exception: If validation forward pass fails
        """
        logger.info("Validating converted checkpoint...")

        try:
            config = model.config
            vocab_size = config.transformer_layer_config.vocab_size
            hidden_size = config.transformer_layer_config.hidden_size
            max_position_embeddings = (
                config.transformer_layer_config.max_position_embeddings
            )

            # Use conservative defaults for batch size and sequence length
            batch_size = 1
            seq_length = min(16, max_position_embeddings)  # Don't exceed max length

            logger.debug(
                f"Running forward pass with batch_size={batch_size}, "
                f"seq_length={seq_length}, vocab_size={vocab_size}, "
                f"hidden_size={hidden_size}"
            )

            model.to(device)  # type: ignore[attr-defined,arg-type]
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(
                device
            )
            hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)
            with torch.no_grad():
                model(input_ids=input_ids, hidden_states=hidden_states)  # type: ignore[operator]
            model.to("cpu")  # type: ignore[attr-defined,arg-type]

            logger.success("Validation forward pass successful")
        except Exception as exception:
            logger.error(f"Validation failed: {exception}")
            raise exception

    def _pretrained_config_from_eagle(self, eagle_config: dict) -> LlamaConfig:
        """
        Create a LlamaConfig for the Eagle model's transformer layer.

        Extracts relevant configuration parameters from the Eagle checkpoint config
        and creates a compatible LlamaConfig for the single transformer layer used
        in Eagle architecture.

        :param eagle_config: Original Eagle checkpoint configuration dictionary
        :return: LlamaConfig configured for Eagle's transformer layer
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

    def _eagle_speculator_config(
        self,
        orig_config: dict,
        fusion_bias: bool,
        layernorms: bool,
    ) -> EagleSpeculatorConfig:
        """
        Build complete EagleSpeculatorConfig from Eagle checkpoint configuration.

        Creates a comprehensive speculator configuration including transformer layer
        config, speculative decoding settings, and feature flags for fusion bias
        and layernorms.

        :param orig_config: Original Eagle checkpoint configuration dictionary
        :param fusion_bias: Whether to enable fusion bias in the speculator
        :param layernorms: Whether to enable extra layernorms in the speculator
        :return: Complete EagleSpeculatorConfig for the converted model
        """
        logger.debug(
            f"Building config with fusion_bias={fusion_bias}, layernorms={layernorms} "
            f"from Eagle checkpoint config: {orig_config}"
        )
        pretrained_config = self._pretrained_config_from_eagle(orig_config)

        return EagleSpeculatorConfig(
            transformer_layer_config=pretrained_config,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        proposal_type="greedy",
                        speculative_tokens=5,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    self.verifier,
                ),
            ),
            layernorms=layernorms,
            fusion_bias=fusion_bias,
        )

    def _should_skip_weight(
        self, weight_name: str, fusion_bias: bool, layernorms: bool
    ) -> bool:
        """
        Determine if a weight should be excluded from the conversion process.

        Checks if a weight from the original Eagle checkpoint should be skipped
        based on its name and the enabled features. Skips embedding tokens, optional
        fusion bias, optional layernorms, and unmapped weights.

        :param weight_name: Name of the weight from original checkpoint
        :param fusion_bias: Whether fusion bias is enabled
        :param layernorms: Whether layernorms are enabled
        :return: True if the weight should be excluded from conversion
        """
        return (
            (weight_name == "embed_tokens.weight")
            or (weight_name == "fc.bias" and not fusion_bias)
            or (weight_name in list(self.LAYERNORM_MAPPINGS.keys()) and not layernorms)
            or (
                not any(
                    weight_name.startswith(prefix) for prefix in self.WEIGHT_MAPPINGS
                )
            )
        )

    def _remap_weight_name(self, weight_name: str) -> str:
        """
        Remap Eagle weight name to Speculators format.

        Transforms weight names from the original Eagle checkpoint format to the
        standardized Speculators format using predefined mappings for fusion layers
        and layernorms.

        :param weight_name: Original weight name from Eagle checkpoint
        :return: Remapped weight name in Speculators format
        :raises ValueError: If weight name doesn't match any known mapping pattern
        """
        mappings = {
            **self.WEIGHT_MAPPINGS,
            **self.LAYERNORM_MAPPINGS,
        }
        for from_mapping, to_mapping in mappings.items():
            if weight_name.startswith(from_mapping):
                return weight_name.replace(from_mapping, to_mapping)

        raise ValueError(
            f"Unexpected weight name format: {weight_name}. "
            "Please check the Eagle checkpoint structure."
        )

    def _eagle_speculator_state_dict(
        self,
        orig_state_dict: dict[str, Tensor],
        fusion_bias: bool,
        layernorms: bool,
    ) -> tuple[dict[str, Tensor], list[str], list[str]]:
        """
        Process and remap all weights from Eagle checkpoint to Speculators format.

        Transforms the complete state dictionary from Eagle format to Speculators
        format, handling weight filtering, name remapping, and tracking of missing
        or extra keys for diagnostic purposes.

        :param orig_state_dict: Original state dictionary from Eagle checkpoint
        :param fusion_bias: Whether fusion bias weights should be included
        :param layernorms: Whether layernorm weights should be included
        :return: Tuple of (converted state dict, missing keys, extra keys)
        """
        logger.debug(
            f"Processing state_dict with fusion_bias={fusion_bias}, "
            f"layernorms={layernorms} from original keys: {orig_state_dict.keys()}"
        )
        converted_state_dict = {}
        missing_keys = []
        extra_keys = []

        for name, tensor in orig_state_dict.items():
            if self._should_skip_weight(name, fusion_bias, layernorms):
                missing_keys.append(name)
                continue

            try:
                new_name = self._remap_weight_name(name)
            except ValueError:
                extra_keys.append(name)
                continue

            converted_state_dict[new_name] = tensor

        logger.debug(
            f"Converted state_dict with {list(converted_state_dict)} weights, "
            f"{list(missing_keys)} missing keys, and {list(extra_keys)} extra keys."
        )

        return converted_state_dict, missing_keys, extra_keys
