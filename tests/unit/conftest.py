"""Shared fixtures for unit tests."""

import pytest
from packaging.version import Version
from transformers import AutoConfig, PretrainedConfig

from tests.conftest import _TRANSFORMERS_VERSION


@pytest.fixture
def qwen3_5_pretrained_config() -> PretrainedConfig:
    if Version("5.2.0") > _TRANSFORMERS_VERSION:
        pytest.skip("qwen3_5 model type requires transformers>=5.2.0")
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    return config.get_text_config()
