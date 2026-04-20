"""Shared pytest configuration and fixtures for all tests."""

import pytest
import torch

# Skip decorators
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2+ GPUs required",
)
