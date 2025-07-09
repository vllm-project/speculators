"""
A module that provides the base class for Speculators model converters handling
the conversion of non-Speculators model checkpoints to the Speculators format.

Classes:
    SpeculatorConverter: An abstract base class for Speculators model converters.

Functions:
    reload_and_populate_converters: Reloads the SpeculatorConverter registry
        and populates it with all registered converter classes.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, Optional, TypeVar, Union

from torch import Tensor, device, nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.utils import ClassRegistryMixin

__all__ = ["SpeculatorConverter"]


ConfigT = TypeVar("ConfigT", bound=SpeculatorModelConfig)
ModelT = TypeVar("ModelT", bound=SpeculatorModel)


class SpeculatorConverter(ABC, Generic[ConfigT, ModelT], ClassRegistryMixin):
    """
    Base class for Speculators model converters.
    This class provides a registry for different conversion algorithms,
    a method to resolve the appropriate converter based on the specified algorithm,
    and the basic structure and methods required for converting a model checkpoint
    to a Speculators model format.
    """

    @classmethod
    def resolve_converter(
        cls,
        algorithm: str,
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        **kwargs,
    ) -> type["SpeculatorConverter"]:
        """
        Return a SpeculatorConverter class based on the specified algorithm.
        If `algorithm` is "auto", it will automatically determine the best
        converter based on the provided model and config utilizing the
        `is_supported` method of each registered converter.

        :param algorithm: The name of the conversion algorithm to use.
            If "auto", it will automatically select the best converter.
        :param model: The model to convert, can be a local path, Hugging Face
            model ID, or a PreTrainedModel instance. Only used for the
            algorithm=auto case.
        :param config: The configuration for the model, can be a local path,
            Hugging Face model ID, or a PretrainedConfig instance.
            Only used for the algorithm=auto case.
        :param verifier: Optional verifier to attach to the converted model.
            Can be a local path to a verifier checkpoint, a Hugging Face model ID,
            or a PreTrainedModel instance.
            Only used for the algorithm=auto case.
        :param kwargs: Additional keyword arguments to pass to the converter's
            `is_supported` method if `algorithm` is "auto".
        :return: An instance of the SpeculatorConverter class for the
            specified algorithm.
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
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        **kwargs,
    ) -> bool:
        """
        Check if the converter supports the given model and config.
        This method should be implemented by each specific converter class.

        :param model: The model to check, can be a local path, Hugging Face
            model ID, or a PreTrainedModel instance.
        :param config: The configuration for the model, can be a local path,
            Hugging Face model ID, or a PretrainedConfig instance.
        :param verifier: Optional verifier to attach to the converted model.
            Can be a local path to a verifier checkpoint, a Hugging Face model ID,
            or a PreTrainedModel instance.
        :param kwargs: Additional keyword arguments for specific checks.
        :return: True if the converter supports the model and config, False otherwise.
        """
        ...

    def __init__(
        self,
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]],
    ):
        """
        Initialize the SpeculatorConverter with the model, config,
        and optional verifier.

        :param model: The model to convert, can be a local path, Hugging Face
            model ID, or a PreTrainedModel instance.
        :param config: The configuration for the model, can be a local path,
            Hugging Face model ID, or a PretrainedConfig instance.
        :param verifier: Optional verifier to attach to the converted model.
            Can be a local path to a verifier checkpoint, a Hugging Face model ID,
            or a PreTrainedModel instance.
        """

        if not model or not config:
            raise ValueError(
                f"Model and config paths must be provided, got {model}, {config}"
            )

        self.model = model
        self.config = config
        self.verifier = verifier

    def __call__(
        self,
        output_path: Optional[Union[str, os.PathLike]] = None,
        validate_device: Optional[Union[str, device, int]] = None,
        verifier_attachment_mode: Literal[
            "detached", "full", "train_only"
        ] = "detached",
    ) -> ModelT:
        """
        Convert the model checkpoint and supporting args for the current instance
        of the SpeculatorConverter to a Speculators model.

        :param output_path: Optional path to save the converted model.
            If provided, the converted model will be saved to this path.
            Otherwise, the model will not be saved to disk.
        :param validate_device: Device to validate the model on after converting.
            If provided, the model will be validated on this device.
            If None, no validation will be performed.
        :param verifier_attachment_mode: The mode for attaching a verifier to the model.
            Can be "detached", "full", or "train_only". Only relevant for the
            usage of the converted instance that is returned.
        :return: The converted Speculators model instance.
        """
        config, state_dict = self.convert_config_state_dict()
        model: ModelT = SpeculatorModel.from_pretrained(  # type: ignore[assignment]
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
        )
        self.attach_verifier(
            model=model,
            verifier_attachment_mode=verifier_attachment_mode,
        )
        if output_path:
            self.save(model, output_path)
        if validate_device:
            self.validate(model, verifier_attachment_mode, validate_device)
        return model

    def attach_verifier(
        self,
        model: ModelT,
        verifier_attachment_mode: Literal["detached", "full", "train_only"],
    ) -> bool:
        """
        Attach a verifier to the model.

        :param model: The converted Speculators model to attach the verifier to.
        :param verifier_attachment_mode: The mode for attaching the verifier.
            Can be "detached", "full", or "train_only".
        :return: True if the verifier was successfully attached,
            False if no verifier was set.
        """
        if self.verifier is None:
            return False

        # ensure verifier is set in the speculator's config
        model.attach_verifier(
            verifier=self.verifier,
            mode=(
                verifier_attachment_mode
                if verifier_attachment_mode != "detached"
                else "train_only"
            ),
        )
        if verifier_attachment_mode == "detached":
            # remove it if input is set to not keep the verifier attached
            model.detach_verifier()

        return True

    def save(self, model: ModelT, output_path: Union[str, os.PathLike]):
        """
        Save the converted model to the specified output path.

        :param model: The converted Speculators model to save.
        :param output_path: The path for the directory where the model will be saved.
            If the path does not exist, it will be created.
        """
        model.save_pretrained(output_path)  # type: ignore[attr-defined]

    @abstractmethod
    def convert_config_state_dict(
        self,
    ) -> tuple[ConfigT, dict[str, Tensor]]:
        """
        Convert the model configuration and state dict to a format suitable for
        the Speculators model.

        :return: A tuple containing the converted configuration and state dict.
            The configuration should be an instance of SpeculatorModelConfig or a
            subclass, and the state dict should be a dictionary mapping parameter names
            to PyTorch tensors.
        """
        ...

    @abstractmethod
    def validate(
        self,
        model: ModelT,
        verifier_attachment_mode: Literal["detached", "full", "train_only"],
        device: Union[str, device, int],
    ):
        """
        Validate the converted model on the specified device.
        This method should ensure that the model is correctly set up and can run
        inference or training on the specified device.

        :param model: The converted Speculators model to validate.
        :param verifier_attachment_mode: The mode that was used to attach the verifier.
            Can be "detached", "full", or "train_only".
        :param device: The device to validate the model on.
            Can be a string (e.g., "cuda", "cpu"), a torch.device instance, or an int
            representing the device index (e.g., 0 for "cuda:0").
        """
        ...


def reload_and_populate_converters():
    """
    Reloads the SpeculatorConverter registry and populates it with all registered
    converter classes. This is useful for dynamically loading converters at runtime.

    :return: None
    """
    SpeculatorConverter.auto_populate_registry()
