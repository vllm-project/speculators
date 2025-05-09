from torch.nn import Module

from speculators.base import DraftModelConfig

__all__ = ["TransformerDrafter"]


class TransformerDrafter(Module):
    @classmethod
    def from_config(cls, config: DraftModelConfig) -> "TransformerDrafter":
        """
        Create a drafter built on top of a Transformer layer as defined in the config.

        :param config: The configuration for the transformer drafter.
        :return: The module instance of the transformer drafter.
        """
        return cls(...)  # Placeholder, need to define and pull args

    def __init__(self, **kwargs):
        """
        Initialize the transformer drafter with the provided arguments.

        :param kwargs: Additional arguments for the transformer drafter.
            Need to define exact arguments for the implementation.
        """
        raise NotImplementedError(
            "TransformerDrafter initialization is not implemented yet."
        )

    @property
    def config(self) -> DraftModelConfig:
        """
        Get the configuration of the transformer drafter.

        :return: The configuration of the transformer drafter.
        """
        return DraftModelConfig(
            type_="transformer",
            inputs=["input_ids"],
            model_config={...},
        )

    def forward(self, **kwargs):
        """
        Run the forward pass of the transformer drafter.

        :param kwargs: Arguments for the forward pass. Still need to define
            exact arguments for the implementation.
        """
        raise NotImplementedError("Forward pass not implemented yet.")
