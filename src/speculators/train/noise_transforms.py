import torch


class TransformTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __call__(self, data):
        for tensor in self.tensors:
            data[tensor] = self.transform(data[tensor])
        return data

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class AddGaussianNoise(TransformTensors):
    def __init__(self, mean=0.0, std=0.2, tensors=("hidden_states",)):
        super().__init__(tensors)
        self.mean = mean
        self.std = std

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class AddUniformNoise(TransformTensors):
    def __init__(self, std=0.2, tensors=("hidden_states",)):
        super().__init__(tensors)
        self.std = std

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + (torch.rand_like(tensor) - 0.5) * self.std
