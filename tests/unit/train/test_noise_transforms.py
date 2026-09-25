"""Unit tests for the tensor noise transforms used during training."""

import pytest
import torch

from speculators.train.noise_transforms import (
    AddGaussianNoise,
    AddUniformNoise,
    TransformTensors,
)


def test_base_transform_raises_not_implemented_error():
    transform = TransformTensors()

    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        transform.transform(torch.zeros(2, 2))


def test_base_call_raises_not_implemented_error():
    transform = TransformTensors()

    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        transform({"hidden_states": torch.zeros(2, 2)})


def test_add_gaussian_noise_transforms_only_configured_keys(seed):
    data = {
        "hidden_states": torch.zeros(4, 8),
        "labels": torch.arange(4),
    }
    original_hidden = data["hidden_states"].clone()
    original_labels = data["labels"].clone()

    result = AddGaussianNoise(std=0.1)(data)

    assert result is data
    assert not torch.equal(data["hidden_states"], original_hidden)
    assert torch.equal(data["labels"], original_labels)


def test_add_gaussian_noise_supports_custom_tensor_keys(seed):
    data = {
        "hidden_states": torch.zeros(2, 2),
        "positions": torch.zeros(2, 2),
    }

    AddGaussianNoise(std=0.1, tensors=("positions",))(data)

    assert torch.equal(data["hidden_states"], torch.zeros(2, 2))
    assert not torch.equal(data["positions"], torch.zeros(2, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_add_gaussian_noise_preserves_shape_dtype_and_device(seed, dtype):
    tensor = torch.randn(3, 5, 7, dtype=dtype)

    noisy = AddGaussianNoise(std=0.05).transform(tensor)

    assert noisy.shape == tensor.shape
    assert noisy.dtype == tensor.dtype
    assert noisy.device == tensor.device


def test_add_gaussian_noise_with_zero_std_is_identity(seed):
    tensor = torch.randn(6, 6)

    assert torch.equal(AddGaussianNoise(std=0.0).transform(tensor), tensor)


def test_add_gaussian_noise_matches_requested_std(seed):
    std = 0.25

    noise = AddGaussianNoise(std=std).transform(torch.zeros(100, 100))

    assert noise.std().item() == pytest.approx(std, rel=0.05)


def test_add_uniform_noise_is_strictly_bounded(seed):
    std = 0.3

    noise = AddUniformNoise(std=std).transform(torch.zeros(10, 10))

    # torch.rand samples lie in [0, 1), so the noise lies in [-std, std).
    assert (noise >= -std).all()
    assert (noise < std).all()
    # The noise should span most of the bounded interval.
    assert noise.abs().max() > 0.9 * std


def test_add_uniform_noise_transforms_only_configured_keys(seed):
    data = {
        "hidden_states": torch.zeros(4, 8),
        "labels": torch.arange(4),
    }
    original_hidden = data["hidden_states"].clone()
    original_labels = data["labels"].clone()

    result = AddUniformNoise(std=0.1)(data)

    assert result is data
    assert not torch.equal(data["hidden_states"], original_hidden)
    assert torch.equal(data["labels"], original_labels)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_add_uniform_noise_preserves_shape_dtype_and_device(seed, dtype):
    tensor = torch.randn(2, 3, 4, dtype=dtype)

    noisy = AddUniformNoise(std=0.1).transform(tensor)

    assert noisy.shape == tensor.shape
    assert noisy.dtype == tensor.dtype
    assert noisy.device == tensor.device


def test_add_uniform_noise_with_zero_std_is_identity(seed):
    tensor = torch.randn(6, 6)

    assert torch.equal(AddUniformNoise(std=0.0).transform(tensor), tensor)
