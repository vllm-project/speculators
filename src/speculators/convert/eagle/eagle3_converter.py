"""
Eagle-3 checkpoint converter with loguru logging.
"""

from pathlib import Path
from typing import Optional, Union

from loguru import logger

from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.eagle3 import Eagle3Speculator


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
