"""
Eagle checkpoint converter with loguru logging.
"""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from loguru import logger
from torch import Tensor
from transformers import LlamaConfig, PreTrainedModel

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
        model: Union[str, os.PathLike],
        config: Union[str, os.PathLike],  # noqa: ARG003
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,  # noqa: ARG003
        fusion_bias: Optional[bool] = None,  # noqa: ARG003
        layernorms: Optional[bool] = None,  # noqa: ARG003
        **kwargs,  # noqa: ARG003
    ) -> bool:
        state_dict = load_model_checkpoint_state_dict(model, keys_only=True)
        has_fc = "fc.bias" in state_dict
        has_layers_0 = any(name.startswith("layers.0.") for name in state_dict)
        has_layers_non_0 = any(
            name.startswith("layers.") and not name.startswith("layers.0.")
            for name in state_dict
        )

        return has_fc and has_layers_0 and not has_layers_non_0

    def __init__(
        self,
        model: Union[str, Path],
        config: Union[str, Path],
        verifier: Optional[Union[str, Path]] = None,
        fusion_bias: Optional[bool] = None,
        layernorms: Optional[bool] = None,
    ):
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

    def validate(
        self,
        model: EagleSpeculator,
        verifier_attachment_mode: Literal["detached", "full", "train_only"],  # noqa: ARG002
        device: Union[str, torch.device, int],
    ):
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

            model.to(device)  # type: ignore[arg-type]
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(
                device
            )
            hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)
            with torch.no_grad():
                model(input_ids=input_ids, hidden_states=hidden_states)
            model.to("cpu")  # type: ignore[arg-type]

            logger.success("Validation forward pass successful")
        except Exception as exception:
            logger.error(f"Validation failed: {exception}")
            raise exception

    def _pretrained_config_from_eagle(self, eagle_config: dict) -> LlamaConfig:
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

    def _eagle_speculator_config(
        self,
        orig_config: dict,
        fusion_bias: bool,
        layernorms: bool,
    ) -> EagleSpeculatorConfig:
        """
        Build a complete EagleSpeculatorConfig from Eagle checkpoint config.

        :param orig_config: Original Eagle checkpoint config
        :param fusion_bias: Whether to enable fusion bias
        :param layernorms: Whether to enable extra layernorms
        :return: Complete Eagle speculator configuration
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
        Determine if a weight should be skipped during conversion.

        :param weight_name: Original weight name
        :param has_layernorms: Whether layernorms are enabled
        :return: True if the weight should be excluded from the output
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
        Remap an Eagle weight name to speculators format.

        :param weight_name: Original weight name
        :return: Remapped weight name
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
        Process and remap all weights from Eagle to speculators format.

        :param orig_state_dict: Original state dict from Eagle checkpoint
        :param fusion_bias: Whether to include fusion bias
        :param layernorms: Whether to include extra layernorms
        :return: Tuple of processed state_dict, missing keys, and extra keys
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
