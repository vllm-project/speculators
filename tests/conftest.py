"""Shared pytest configuration and fixtures for all tests."""

import subprocess
from importlib.metadata import version as pkg_version

import pytest
import torch
from packaging.version import Version

from tests.e2e.utils import VLLM_PYTHON


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
    return pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="2+ GPUs required",
    )(fn)


_TRANSFORMERS_VERSION = Version(pkg_version("transformers"))


def requires_transformers_version(min_version: str):
    return pytest.mark.skipif(
        Version(min_version) > _TRANSFORMERS_VERSION,
        reason=(
            f"transformers>={min_version} required (installed: {_TRANSFORMERS_VERSION})"
        ),
    )


def requires_vllm_version(min_version: str):
    out = subprocess.check_output(  # noqa: S603
        [
            VLLM_PYTHON,
            "-c",
            "from importlib.metadata import version; print(version('vllm'))",
        ],
        text=True,
    )
    installed = Version(out.strip())
    return pytest.mark.skipif(
        Version(min_version) > installed,
        reason=f"vllm>={min_version} required (installed: {installed})",
    )
