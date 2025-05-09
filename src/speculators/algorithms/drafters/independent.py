from pathlib import Path
from typing import Union

from torch.nn import Module

from speculators.base import Drafter, DraftModelConfig
from speculators.utils import load_model

__all__ = ["IndependentDrafter"]


class IndependentDrafter(Drafter):
    @classmethod
    def from_config(
        cls,
        config: DraftModelConfig,  # noqa: ARG003
    ) -> "IndependentDrafter":
        """
        Create an independent drafter (separate model) from the provided config.

        :param config: The configuration for the independent drafter.
        :return: The module instance of the independent drafter.
        """
        # Placeholder, need to define and pull args from config
        return cls(...)  # type: ignore[arg-type,call-arg]

    def __init__(self, source: Union[str, Path, Module]):
        """
        Initialize the independent drafter with the provided source.

        :param source: The source of the independent drafter, which can be a path
            to a model file or a pre-trained model.
        """
        self.model = load_model(source)

    @property
    def config(self) -> DraftModelConfig:
        """
        Get the configuration of the independent drafter.

        :return: The configuration of the independent drafter.
        """
        return DraftModelConfig(
            type_="independent",
            inputs=["input_ids"],
            config={...},  # type: ignore[arg-type,call-arg]
        )

    def forward(self, **kwargs):
        """
        Run the forward pass of the independent drafter.
        :param kwargs: Arguments for the forward pass. Still need to define
            exact arguments for the implementation.
        """
        return self.model(**kwargs)
