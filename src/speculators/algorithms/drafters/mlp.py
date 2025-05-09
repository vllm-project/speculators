from speculators.base import Drafter, DraftModelConfig

__all__ = ["MLPDrafter"]


class MLPDrafter(Drafter):
    @classmethod
    def from_config(cls, config: DraftModelConfig) -> "MLPDrafter":
        """
        Create a MLP drafter (multi-layer perceptron / feed-forward neural network)
        from the provided config.

        :param config: The configuration for the MLP drafter.
        :return: The module instance of the MLP drafter.
        """
        return cls(...)  # Placeholder, need to define and pull args from config

    def __init__(self, **kwargs):
        """
        Initialize the MLP drafter with the provided arguments.

        :param kwargs: Additional arguments for the MLP drafter.
            Need to define exact arguments for the implementation.
        """
        raise NotImplementedError("MLPDrafter initialization is not implemented yet.")

    @property
    def config(self) -> DraftModelConfig:
        """
        Get the configuration of the MLP drafter.

        :return: The configuration of the MLP drafter.
        """
        return DraftModelConfig(type_="ffn", inputs=["input_ids"], model_config={...})

    def forward(self, **kwargs):
        """
        Run the forward pass of the MLP drafter.

        :param kwargs: Arguments for the forward pass. Still need to define
            exact arguments for the implementation.
        """
        raise NotImplementedError("Forward pass not implemented yet.")
