"""
EAGLE3 checkpoint converter for Speculators model format.

This module provides the Eagle3SpeculatorConverter class for transforming EAGLE3-style
speculative decoding checkpoints from research repositories into the standardized
Speculators format. The converter handles automatic feature detection, weight remapping,
configuration translation, vocabulary mapping, and validation through forward passes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import torch
from loguru import logger
from torch import Tensor, nn
from transformers import LlamaConfig, PretrainedConfig, PreTrainedModel

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.converters.base import SpeculatorConverter
from speculators.models.eagle3 import Eagle3Speculator, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils import (
    load_model_checkpoint_config_dict,
    load_model_checkpoint_state_dict,
)

__all__ = ["Eagle3SpeculatorConverter"]


@SpeculatorConverter.register(["eagle3"])
class Eagle3SpeculatorConverter(
    SpeculatorConverter[Eagle3SpeculatorConfig, Eagle3Speculator]
):
    """
    Converter for EAGLE3 research checkpoint format to Speculators format.

    This converter transforms EAGLE3-style speculative decoding checkpoints into the
    standardized Speculators format, handling weight remapping, configuration
    translation, vocabulary mapping, and feature detection. It supports the EAGLE3
    architecture with cross-tokenizer vocabulary mapping and norm_before_residual
    functionality.

    Example:
        ::

        from speculators.convert.converters import Eagle3SpeculatorConverter

        converter = Eagle3SpeculatorConverter(
            model="path/to/eagle3_checkpoint",
            config="path/to/config.json",
            verifier="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        converted_model = converter(output_path="./output", validate_device="cuda")

    :cvar parameter_mappings: Parameter name mappings from EAGLE3 to Speculators format
    :cvar keep_parameters: Parameters to keep from original EAGLE3 checkpoint without
        remapping
    """

    parameter_mappings: Annotated[
        dict[str, str],
        "Parameter name mappings from EAGLE3 checkpoint format to Speculators format",
    ] = {"midlayer.": "layers.0."}
    keep_parameters: Annotated[
        list[str],
        "Parameters to keep from the original EAGLE3 checkpoint without remapping",
    ] = ["d2t", "t2d", "embed_tokens.", "fc.", "lm_head.", "norm."]

    @classmethod
    def is_supported(
        cls,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        norm_before_residual: bool | None = None,
        **kwargs,
    ) -> bool:
        """
        Check if the provided model checkpoint is supported by this converter.

        :param model: Model checkpoint path or instance to validate
        :param config: Model configuration (unused for EAGLE3 detection)
        :param verifier: Optional verifier model (unused for EAGLE3 detection)
        :param norm_before_residual: Optional norm setting (unused for EAGLE3 detection)
        :param kwargs: Additional arguments passed to state dict loader for transformers
            loading functions
        :return: True if the model follows EAGLE3 architecture pattern
        """
        _ = (config, verifier, norm_before_residual)  # remove linting error
        state_dict = load_model_checkpoint_state_dict(model, **kwargs)
        supported_count = sum([cls.is_supported_param(name)[0] for name in state_dict])

        # Ensure minimum number of params in state_dict and then True if all supported
        return supported_count > len(cls.keep_parameters) and supported_count == len(
            state_dict
        )

    @classmethod
    def is_supported_param(cls, name: str) -> tuple[bool, str]:
        """
        Check if a parameter name is supported and return its mapped name.

        :param name: Parameter name to check for support
        :return: Tuple of (is_supported, mapped_name)
        """
        if name.startswith("model."):
            name = name[len("model.") :]

        for remap_prefix, remap_replace in cls.parameter_mappings.items():
            if name.startswith(remap_prefix):
                return True, name.replace(remap_prefix, remap_replace)

        for keep_name in cls.keep_parameters:
            if name.startswith(keep_name):
                return True, name

        return False, name

    def __init__(
        self,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        norm_before_residual: bool | None = None,
    ):
        """
        Initialize the EAGLE3 converter.

        :param model: Model checkpoint path or instance to convert
        :param config: Model configuration path or instance
        :param verifier: Optional verifier model path or instance for speculative
            decoding
        :param norm_before_residual: Whether to apply normalization before residual,
            automatically detected from configuration if None
        """
        super().__init__(
            model=model,
            config=config,
            verifier=verifier,
        )
        self.norm_before_residual = norm_before_residual

    def convert_config_state_dict(
        self,
    ) -> tuple[Eagle3SpeculatorConfig, dict[str, Tensor]]:
        """
        Convert EAGLE3 checkpoint configuration and state dict to Speculators format.

        :return: Tuple of converted configuration and remapped state dictionary
        """
        logger.info(
            f"Converting EAGLE3 checkpoint at model: {self.model} and "
            f"config: {self.config} to speculators format..."
        )
        orig_state_dict = load_model_checkpoint_state_dict(self.model)
        orig_config = load_model_checkpoint_config_dict(self.config)

        if "t2d" in orig_state_dict:
            # Patch: ensure target_vocab_size matches t2d tensor shape
            orig_config["target_vocab_size"] = orig_state_dict["t2d"].shape[0]
            logger.info(
                f"Set target_vocab_size to {orig_config['target_vocab_size']} "
                "based on t2d tensor shape"
            )

        converted_config = self._speculator_config(orig_config)
        config_str = (
            self.config if isinstance(self.config, (str, Path)) else type(self.config)
        )
        logger.info(f"Converted EAGLE3 config from {config_str} to speculators format")

        converted_state_dict, extra = self._speculator_state_dict(orig_state_dict)
        model_str = (
            self.model if isinstance(self.model, (str, Path)) else type(self.model)
        )
        logger.info(
            f"Converted Eagle3 state_dict from {model_str} to speculators format"
        )
        if extra:
            logger.warning(f"Extra keys in converted state_dict: {extra}")

        return converted_config, converted_state_dict

    def validate(self, model: Eagle3Speculator, device: str | torch.device | int):
        """
        Validate the converted model by running a forward pass with test data.

        :param model: The converted Eagle3Speculator model to validate
        :param device: Device for validation (string, torch.device, or device index)
        :raises Exception: If validation forward pass fails
        """
        logger.info("Validating converted checkpoint...")

        try:
            config = model.config
            draft_vocab_size = config.draft_vocab_size
            hidden_size = config.transformer_layer_config.hidden_size

            # Use conservative defaults for batch size and sequence length
            batch_size = 1
            seq_length = 16

            model.to(device)
            input_ids = torch.randint(0, draft_vocab_size, (batch_size, seq_length)).to(
                device
            )
            hidden_states = torch.randn(batch_size, seq_length, 2 * hidden_size).to(
                device
            )
            with torch.no_grad():
                model(input_ids=input_ids, hidden_states=hidden_states)
            model.to("cpu")

            logger.success("Validation forward pass successful")
        except Exception as exception:
            logger.error(f"Validation failed: {exception}")
            raise exception

    def _speculator_config(
        self,
        orig_config: dict,
    ) -> Eagle3SpeculatorConfig:
        logger.debug(
            f"Building config with norm_before_residual={self.norm_before_residual} "
            f"from Eagle3 checkpoint config: {orig_config}"
        )
        pretrained_config = self._pretrained_config(
            orig_config,
            verifier_config=(
                load_model_checkpoint_config_dict(self.verifier)
                if self.verifier
                else {}
            ),
        )

        return Eagle3SpeculatorConfig(
            transformer_layer_config=pretrained_config,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        proposal_type="greedy",
                        speculative_tokens=5,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(self.verifier),
            ),
            draft_vocab_size=orig_config.get("draft_vocab_size", 32000),
            norm_before_residual=self.norm_before_residual,
            target_hidden_size=orig_config.get("target_hidden_size"),
        )

    def _pretrained_config(
        self, orig_config: dict, verifier_config: dict
    ) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=orig_config.get("target_vocab_size", 128000),
            hidden_size=orig_config.get("hidden_size", 4096),
            intermediate_size=orig_config.get("intermediate_size", 11008),
            num_hidden_layers=1,
            num_attention_heads=orig_config.get("num_attention_heads", 32),
            num_key_value_heads=orig_config.get("num_key_value_heads", 8),
            hidden_act=orig_config.get("hidden_act", "silu"),
            # Ensure max_position_embeddings match between Eagle3 and target configs
            max_position_embeddings=max(
                orig_config.get("max_position_embeddings", 4096),
                verifier_config.get("max_position_embeddings", 4096),
            ),
            initializer_range=orig_config.get("initializer_range", 0.02),
            rms_norm_eps=orig_config.get("rms_norm_eps", 1e-6),
            use_cache=True,
            attention_bias=orig_config.get("attention_bias", False),
            rope_theta=orig_config.get("rope_theta", 10000.0),
            mlp_bias=orig_config.get("mlp_bias", False),
            tie_word_embeddings=False,
        )

    def _speculator_state_dict(
        self,
        orig_state_dict: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], list[str]]:
        logger.debug(
            f"Processing Eagle3 state_dict from original keys: {orig_state_dict.keys()}"
        )
        converted_state_dict = {}
        extra_keys = []

        # Remap params from original EAGLE3 state_dict according to defined mappings
        for name, tensor in orig_state_dict.items():
            keep, converted_name = self.is_supported_param(name)
            if keep:
                converted_state_dict[converted_name] = tensor
            else:
                extra_keys.append(name)

        if "embed_tokens.weight" not in converted_state_dict:
            # Load embed_tokens.weight from verifier if not present in EAGLE3 checkpoint
            if not self.verifier:
                raise RuntimeError(
                    "embed_tokens.weight not found in EAGLE3 checkpoint and no "
                    "verifier provided to source it from"
                )

            verifier_state_dict = load_model_checkpoint_state_dict(self.verifier)
            if "model.embed_tokens.weight" in verifier_state_dict:
                embed_name = "model.embed_tokens.weight"
            elif "embed_tokens.weight" in verifier_state_dict:
                embed_name = "embed_tokens.weight"
            else:
                raise RuntimeError(
                    "embed_tokens.weight not found in EAGLE3 checkpoint or verifier"
                )
            converted_state_dict["embed_tokens.weight"] = verifier_state_dict[
                embed_name
            ]
            logger.info(
                f"Added embed_tokens.weight from verifier {embed_name} "
                f"with shape f{verifier_state_dict[embed_name].shape}"
            )
            del verifier_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.debug(
            f"Converted state_dict with {list(converted_state_dict)} weights, "
            f"and {list(extra_keys)} extra keys."
        )

        return converted_state_dict, extra_keys
