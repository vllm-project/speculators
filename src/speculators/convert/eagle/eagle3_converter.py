"""
Eagle-3 checkpoint converter with loguru logging.
"""

import os
from pathlib import Path
from typing import Optional, Union

from torch import nn
from loguru import logger
from transformers import LlamaConfig, PreTrainedModel

from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.convert.eagle.base import SpeculatorConverter


@SpeculatorConverter.register("eagle3")
class Eagle3Converter(SpeculatorConverter):
    """
    Converter for Eagle-3 checkpoints to speculators format.
    Supports automatic feature detection, weight remapping, and optional validation.
    """

    def __init__(
        self,
        model: Union[Path, PreTrainedModel, nn.Module, str],
        verifier: Union[str, os.PathLike, PreTrainedModel],
        output_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):

        super().__init__(
            model=model,
            verifier=verifier,
            output_path=output_path,
            model_type="eagle3",
            cache_dir=cache_dir,
        )

    def convert(
        self,
        validate: bool = True,
        norm_before_residual: bool = False,
    ) -> None:
        # Patch: ensure target_vocab_size matches t2d tensor shape
        self.config["target_vocab_size"] = self.weights["t2d"].shape[0]

        config = self._build_eagle3_speculator_config(
            norm_before_residual,
        )

        saved_path = self._save_converted_checkpoint(config, self.weights, self.output_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate_converted_checkpoint(saved_path, self.verifier)

    def _build_eagle3_speculator_config(
        self,
        norm_before_residual: bool = False,
    ) -> Eagle3SpeculatorConfig:
        """
        Build a complete EagleSpeculatorConfig from Eagle checkpoint config.

        :return: Complete Eagle speculator configuration
        """
        transformer_config = self._create_transformer_config()
        speculators_config = self._create_speculator_config()

        return Eagle3SpeculatorConfig(
            transformer_layer_config=transformer_config,
            speculators_config=speculators_config,
            draft_vocab_size=self.config.get("draft_vocab_size", 32000),
            norm_before_residual=norm_before_residual,
            target_hidden_size=self.config.get("target_hidden_size"),
        )

    def _create_transformer_config(self) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=self.config.get("target_vocab_size", 128000),
            hidden_size=self.config.get("hidden_size", 4096),
            intermediate_size=self.config.get("intermediate_size", 11008),
            num_hidden_layers=1,
            num_attention_heads=self.config.get("num_attention_heads", 32),
            num_key_value_heads=self.config.get("num_key_value_heads", 8),
            hidden_act=self.config.get("hidden_act", "silu"),
            max_position_embeddings=self.config.get("max_position_embeddings", 4096),
            initializer_range=self.config.get("initializer_range", 0.02),
            rms_norm_eps=self.config.get("rms_norm_eps", 1e-6),
            use_cache=True,
            attention_bias=self.config.get("attention_bias", False),
            rope_theta=self.config.get("rope_theta", 10000.0),
            mlp_bias=self.config.get("mlp_bias", False),
            tie_word_embeddings=False,
        )
