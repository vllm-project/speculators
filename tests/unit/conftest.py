"""Shared fixtures for unit tests."""

import pytest
from transformers import AutoConfig, PretrainedConfig


@pytest.fixture
def qwen3_5_pretrained_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
    return config.get_text_config()
