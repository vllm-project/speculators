from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from transformers import PretrainedConfig

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils import ClassRegistryMixin


class SpeculatorConverter(ClassRegistryMixin):
    def __init__(
        self,
        model: Union[str, Path],
        verifier: str,
        output_path: Optional[Union[str, Path]] = None,
        model_type: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        :param model: Path or ID of Eagle checkpoint (local or from HuggingFace Hub)
        :param verifier: Verifier model name or path
        :param output_path: Where to save the converted model
        :param model_type: Model algorithm type (e.g., 'eagle', 'eagle3')
        :param cache_dir: Optional cache directory for downloads
        """
        if not model:
            raise ValueError("Model must be provided")
        if not model_type:
            raise ValueError("model_type must be specified for the converter")

        self.model = model
        self.verifier = verifier
        self.output_path = Path(output_path) if output_path else Path.cwd()
        self.model_type = model_type

        logger.info(f"Converting Eagle checkpoint: {model}")
        local_checkpoint_path = ensure_checkpoint_is_local(model, cache_dir)

        self.config = load_checkpoint_config(local_checkpoint_path)
        self.weights = load_checkpoint_weights(local_checkpoint_path)
        logger.info(f"Loaded {len(self.weights)} weights")

    def _create_speculator_config(
        self,
    ) -> SpeculatorsConfig:
        """
        Build SpeculatorsConfig for Eagle variants based on self.model_type.

        :return: SpeculatorsConfig instance configured for the model type
        """
        proposal_config = GreedyTokenProposalConfig(
            proposal_type="greedy",
            speculative_tokens=5,
        )

        verifier_config = self._create_verifier_config()

        return SpeculatorsConfig(
            algorithm=self.model_type,  # e.g. "eagle2" or "eagle3"
            proposal_methods=[proposal_config],
            default_proposal_method="greedy",
            verifier=verifier_config,
        )

    def _create_verifier_config(self) -> VerifierConfig:
        """
        Create a VerifierConfig from a base model name or path.

        :return: VerifierConfig object with model details for the verifier
        """
        config_dict, _ = PretrainedConfig.get_config_dict(self.verifier)
        return VerifierConfig(
            name_or_path=self.verifier,
            architectures=config_dict.get("architectures", ["LlamaForCausalLM"]),
        )

    def _save_converted_checkpoint(
        self,
        config,
        weights: dict[str, torch.Tensor],
        output_dir: Union[str, Path],
    ) -> Path:
        """
        Instantiate the model from registry, load weights, and save checkpoint.

        :param config: Model configuration object for the converted model
        :param weights: Dictionary mapping parameter names to tensors
        :param output_dir: Directory path to save the converted checkpoint
        :return: Path to the saved checkpoint directory
        :raises ValueError: If model class for self.model_type is not registered
        """
        if SpeculatorConverter.registry is None:
            raise RuntimeError("Converter registry is not initialized")
        model_class = SpeculatorModel.registry.get(self.model_type)  # type: ignore[union-attr]

        model = model_class(  # type: ignore[misc]
            config=config,
            verifier=None,
            verifier_attachment_mode="detached",
        )
        model.load_state_dict(weights, strict=False)
        model.save_pretrained(str(output_dir))
        return Path(output_dir)

    def _validate_converted_checkpoint(
        self, checkpoint_path: Path, verifier_model: str
    ) -> None:
        """
        Validate the converted checkpoint by loading and instantiating with verifier.

        :param checkpoint_path: Path to the converted checkpoint directory
        :param verifier_model: Verifier model name or path used for validation
        :raises ValueError: If model class for self.model_type is not registered
        :raises Exception: Propagates any errors raised during model loading
        """
        if SpeculatorConverter.registry is None:
            raise RuntimeError("Converter registry is not initialized")
        model_class = SpeculatorModel.registry.get(self.model_type)  # type: ignore[union-attr]

        logger.info("Validating converted checkpoint...")
        try:
            model_class.from_pretrained(  # type: ignore[union-attr]
                checkpoint_path,
                verifier=verifier_model,
                verifier_attachment_mode="detached",
            )
            logger.success("Validation succeeded")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
