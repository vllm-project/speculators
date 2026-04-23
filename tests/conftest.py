"""Shared pytest configuration and fixtures for all tests."""

import pytest
import torch


@pytest.fixture
def seed():
    torch.manual_seed(42)
    yield 42  # noqa: PT022


# Skip decorators
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
def requires_multi_gpu(fn):
    fn = pytest.mark.multi_gpu(fn)
    fn = pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="2+ GPUs required",
    )(fn)
    return fn
