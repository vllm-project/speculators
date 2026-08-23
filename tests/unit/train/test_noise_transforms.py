"""Unit tests for the hidden-state noise transforms."""

import pytest
import torch

from speculators.train.noise_transforms import AddGaussianNoise, AddUniformNoise


@pytest.mark.parametrize("cls", [AddGaussianNoise, AddUniformNoise])
def test_zero_std_is_identity_and_consumes_no_rng(cls):
    """With std=0 the data is returned unchanged and the generator does not advance."""
    data = {"hidden_states": torch.arange(6.0).reshape(2, 3)}
    before = torch.get_rng_state()
    out = cls(std=0.0)(dict(data))
    assert torch.equal(out["hidden_states"], data["hidden_states"])
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("cls", [AddGaussianNoise, AddUniformNoise])
def test_positive_std_still_adds_noise(cls):
    """Non-zero std keeps the existing behaviour."""
    data = {"hidden_states": torch.zeros(2, 3)}
    out = cls(std=0.1)(dict(data))
    assert not torch.equal(out["hidden_states"], data["hidden_states"])
