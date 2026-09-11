"""Noise transforms for augmenting tensors (e.g. hidden states) during training."""

import torch

__all__ = [
    "AddGaussianNoise",
    "AddUniformNoise",
    "TransformTensors",
]


class TransformTensors:
    """Base class for noise transforms applied to named tensors of a batch.

    Subclasses must override :meth:`transform` to define the noise operation
    applied to each of the configured tensors.
    """

    def __init__(
        self, std: float = 0.05, tensors: tuple[str, ...] = ("hidden_states",)
    ):
        """Initialize the transform.

        Args:
            std: Scale of the noise applied to the configured tensors.
            tensors: Keys of the batch entries to transform.
        """
        self.tensors = tensors
        self.std = std

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the transform to each configured tensor of the batch.

        Args:
            data: Mapping of batch entries to tensors. Entries listed in
                ``self.tensors`` are replaced with their transformed values.

        Returns:
            The batch mapping with the configured tensors transformed.
        """
        for tensor in self.tensors:
            data[tensor] = self.transform(data[tensor])
        return data

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise to a single tensor.

        Args:
            tensor: The tensor to transform.

        Returns:
            The transformed tensor.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")


class AddGaussianNoise(TransformTensors):
    """Add zero-mean Gaussian noise to the configured tensors."""

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with standard deviation ``self.std`` to a tensor.

        Args:
            tensor: The tensor to transform.

        Returns:
            A new tensor with element-wise Gaussian noise added.
        """
        return tensor + torch.randn_like(tensor) * self.std


class AddUniformNoise(TransformTensors):
    """Add uniform noise to the configured tensors."""

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add uniform noise bounded by ``self.std`` to a tensor.

        Args:
            tensor: The tensor to transform.

        Returns:
            A new tensor with element-wise uniform noise added, where each
            noise sample lies within ``[-self.std, self.std)``.
        """
        return tensor + 2 * (torch.rand_like(tensor) - 0.5) * self.std
