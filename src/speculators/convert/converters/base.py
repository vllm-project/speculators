"""
Abstract base converter for transforming external checkpoints to Speculators format.

This module provides the registry-based converter architecture for standardizing
external research model checkpoints into the Speculators format. The converter
system supports automatic algorithm detection, extensible conversion workflows,
and validation for various speculative decoding implementations.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from torch import Tensor, device, nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.utils import RegistryMixin

__all__ = ["ConfigT", "ModelT", "SpeculatorConverter"]


ConfigT = TypeVar("ConfigT", bound=SpeculatorModelConfig)
"""Generic type variable for speculator model configs"""
ModelT = TypeVar("ModelT", bound=SpeculatorModel)
"""Generic type variable for speculator models"""


class SpeculatorConverter(ABC, RegistryMixin, Generic[ConfigT, ModelT]):
    """
    Abstract base converter for transforming external checkpoints to Speculators format.

    Provides a registry-based system for algorithm-specific converters with automatic
    detection capabilities. The converter handles the complete transformation pipeline
    including configuration translation, state dict conversion, model instantiation,
    and validation for various speculative decoding implementations.

    Example:
    ::
        # Resolve converter automatically
        converter_cls = SpeculatorConverter.resolve_converter(
            algorithm="auto", model="path/to/model", config="path/to/config"
        )

        # Create and execute conversion
        converter = converter_cls(model, config, verifier=None)
        model = converter(output_path="converted_model", validate_device="cuda")
    """

    @classmethod
    def resolve_converter(
        cls,
        algorithm: str,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        **kwargs,
    ) -> type[SpeculatorConverter]:
        """
        Resolve the appropriate converter class for the specified algorithm.

        Supports automatic algorithm detection when algorithm="auto" by testing each
        registered converter's `is_supported` method against the provided inputs.

        :param algorithm: Conversion algorithm name or "auto" for automatic detection
        :param model: Model to convert (path, HF model ID, or PreTrainedModel instance)
        :param config: Model configuration (path, HF model ID, or PretrainedConfig)
        :param verifier: Optional verifier model for speculative decoding attachment
        :param kwargs: Additional arguments passed to `is_supported` for auto detection
        :return: Converter class for the specified or detected algorithm
        :raises ValueError: If algorithm is not registered or no supported converter
            found
        """
        if cls.registry is None:
            raise ValueError(
                "No converters registered. Please ensure that the SpeculatorConverter "
                "subclass has registered converters using the @register decorator."
            )

        algorithm = algorithm.lower()

        if algorithm != "auto":
            if algorithm not in cls.registry:
                raise ValueError(
                    f"Algorithm '{algorithm}' is not registered. "
                    f"Available algorithms: {', '.join(cls.registry.keys())}"
                )
            return cls.registry[algorithm]  # type: ignore[return-value]

        for _, converter in cls.registry.items():
            if converter.is_supported(model, config, verifier, **kwargs):
                return converter  # type: ignore[return-value]

        raise ValueError(
            f"No supported converter found for model {model} with config {config}. "
            f"Available algorithms: {', '.join(cls.registry.keys())}"
        )

    @classmethod
    @abstractmethod
    def is_supported(
        cls,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        **kwargs,
    ) -> bool:
        """
        Check if this converter supports the given model and configuration.

        :param model: Model to check (path, HF model ID, or PreTrainedModel instance)
        :param config: Model configuration (path, HF model ID, or PretrainedConfig)
        :param verifier: Optional verifier model for compatibility validation
        :param kwargs: Additional arguments for algorithm-specific checks
        :return: True if the converter supports the model and config
        """
        ...

    def __init__(
        self,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None,
    ):
        """
        Initialize the converter with model, configuration, and optional verifier.

        :param model: Model to convert (path, HF model ID, or PreTrainedModel instance)
        :param config: Model configuration (path, HF model ID, or PretrainedConfig)
        :param verifier: Optional verifier model for speculative decoding attachment
        :raises ValueError: If model or config is None or empty
        """

        if model is None or config is None or model == "" or config == "":
            raise ValueError(
                f"Model and config paths must be provided, got {model}, {config}"
            )

        self.model = model
        self.config = config
        self.verifier = verifier

    def __call__(
        self,
        output_path: str | os.PathLike | None = None,
        validate_device: str | device | int | None = None,
    ) -> ModelT:
        """
        Execute the complete conversion pipeline to Speculators format.

        Converts configuration and state dict, instantiates the model, optionally
        saves to disk, and validates on the specified device.

        :param output_path: Optional directory path to save the converted model
        :param validate_device: Optional device for post-conversion validation
        :return: Converted Speculators model instance
        """
        config, state_dict = self.convert_config_state_dict()
        model: ModelT = SpeculatorModel.from_pretrained(  # type: ignore[assignment]
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
            verifier=self.verifier,
            verifier_attachment_mode="full",
        )
        if output_path:
            self.save(model, output_path)
        if validate_device:
            self.validate(model, validate_device)
        return model

    def save(self, model: ModelT, output_path: str | os.PathLike):
        """
        Save the converted model to the specified directory.

        :param model: Converted Speculators model to save
        :param output_path: Directory path where the model will be saved
        """
        model.save_pretrained(output_path)  # type: ignore[attr-defined]

    @abstractmethod
    def convert_config_state_dict(self) -> tuple[ConfigT, dict[str, Tensor]]:
        """
        Convert model configuration and state dict to Speculators format.

        :return: Tuple of (converted configuration, converted state dict)
        """
        ...

    @abstractmethod
    def validate(self, model: ModelT, device: str | device | int):
        """
        Validate the converted model on the specified device.

        :param model: Converted Speculators model to validate
        :param device: Device for validation (string, torch.device, or device index)
        """
        ...
