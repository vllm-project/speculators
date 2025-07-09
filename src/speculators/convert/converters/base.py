import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, Optional, TypeVar, Union

from torch import Tensor, device
from transformers import PreTrainedModel

from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.utils import ClassRegistryMixin

__all__ = ["SpeculatorConverter"]


ConfigT = TypeVar("ConfigT", bound=SpeculatorModelConfig)
ModelT = TypeVar("ModelT", bound=SpeculatorModel)


class SpeculatorConverter(ABC, Generic[ConfigT, ModelT], ClassRegistryMixin):
    @classmethod
    def resolve_converter(
        cls,
        algorithm: str,
        model: Union[str, os.PathLike],
        config: Union[str, os.PathLike],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        **kwargs,
    ) -> "SpeculatorConverter":
        """
        Match the appropriate conversion algorithm based on the model and config.
        This method iterates through the registered converters and checks if they
        support the given model and config.

        :param model: Path to the model or HuggingFace model ID
        :param config: Path to the model configuration or HuggingFace model ID
        :param verifier: Optional verifier model or path
        :param kwargs: Additional keyword arguments for converter-specific checks
        :return: The name of the matching algorithm
        :raises ValueError: If no matching converter is found
        """
        algorithm = algorithm.lower()

        if algorithm != "auto":
            if algorithm not in cls.registry:
                raise ValueError(
                    f"Algorithm '{algorithm}' is not registered. "
                    f"Available algorithms: {', '.join(cls.registry.keys())}"
                )
            return cls.registry[algorithm]

        for algorithm, converter in cls.registry.items():
            if converter.is_supported(model, config, verifier, **kwargs):
                return algorithm

        raise ValueError(
            f"No supported converter found for model {model} with config {config}. "
            f"Available algorithms: {', '.join(cls.registry.keys())}"
        )

    @classmethod
    @abstractmethod
    def is_supported(
        cls,
        model: Union[str, os.PathLike],
        config: Union[str, os.PathLike],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        **kwargs,
    ) -> bool: ...

    def __init__(
        self,
        model: Union[str, os.PathLike],
        config: Union[str, os.PathLike],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]],
    ):
        if not model or not config:
            raise ValueError(
                f"Model and config paths must be provided, got {model}, {config}"
            )

        self.model = Path(model)
        self.config = Path(config)
        self.verifier = verifier

        if not self.model.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model}")

        if not self.config.exists():
            raise FileNotFoundError(f"Config path does not exist: {self.config}")

    def __call__(
        self,
        output_path: Optional[Union[str, os.PathLike]] = None,
        validate_device: Optional[Union[str, device, int]] = None,
        verifier_attachment_mode: Literal[
            "detached", "full", "train_only"
        ] = "detached",
    ) -> ModelT:
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
        model.save_pretrained(output_path)

    @abstractmethod
    def convert_config_state_dict(
        self,
    ) -> tuple[ConfigT, dict[str, Tensor]]: ...

    @abstractmethod
    def validate(
        self,
        model: ModelT,
        verifier_attachment_mode: Literal["detached", "full", "train_only"],
        device: Union[str, device, int],
    ): ...
