import torch


class TransformTensors:
    def __init__(self, std=0.05, tensors=("hidden_states",)):
        self.tensors = tensors
        self.std = std

    def __call__(self, data):
        for tensor in self.tensors:
            data[tensor] = self.transform(data[tensor])
        return data

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class AddGaussianNoise(TransformTensors):
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


class AddUniformNoise(TransformTensors):
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 2 * (torch.rand_like(tensor) - 0.5) * self.std
