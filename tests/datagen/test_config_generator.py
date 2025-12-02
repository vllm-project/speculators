"""
Unit tests for the config_generator module.
"""

from __future__ import annotations

from unittest import mock

import pytest

from speculators.data_generation.config_generator import DataGenerationConfig

TRAIN_DATA_PATH = "sharegpt"
SEQ_LENGTH = 2048
CACHE_DIR = "/cache"
NUM_SAMPLES = 100


@pytest.fixture
def mock_vllm_generator():
    """Mock VllmHiddenStatesGenerator for testing."""
    generator = mock.MagicMock()
    generator.model_path = "meta-llama/Llama-3.1-8B-Instruct"
    generator.layer_ids = [2, 16, 29, 31]
    generator.tensor_parallel_size = 1
    generator.vllm_config.model_config.max_model_len = 2048
    generator.vllm_config.cache_config.gpu_memory_utilization = 0.8
    return generator


def _create_model_config_fixture(hidden_size=None, text_config_hidden_size=None):
    """Factory for creating model config mocks with different hidden_size."""
    config = mock.MagicMock()
    config.hidden_size = hidden_size

    if text_config_hidden_size is not None:
        config.text_config = mock.MagicMock()
        config.text_config.hidden_size = text_config_hidden_size
    else:
        config.text_config = None

    return config


@pytest.fixture
def model_config_direct():
    """Create AutoConfig mock with direct hidden_size attribute."""
    config = _create_model_config_fixture(hidden_size=4096)
    with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config):
        yield


@pytest.fixture
def model_config_text_config():
    """Create AutoConfig mock with text_config.hidden_size attribute."""
    config = _create_model_config_fixture(text_config_hidden_size=2048)
    with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config):
        yield


@pytest.fixture
def model_config_missing():
    """Create AutoConfig mock with no hidden_size attribute."""
    config = _create_model_config_fixture()
    with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config):
        yield


def create_config(generator):
    """Helper to create config with consistent test parameters."""
    return DataGenerationConfig.from_generator(
        generator=generator,
        train_data_path=TRAIN_DATA_PATH,
        seq_length=SEQ_LENGTH,
        cache_dir=CACHE_DIR,
        num_samples=NUM_SAMPLES,
    )


@pytest.mark.smoke
def test_config_from_generator_with_direct_hidden_size(
    mock_vllm_generator, model_config_direct
):
    """Test config extraction with direct hidden_size attribute."""
    config = create_config(mock_vllm_generator)
    assert config.model.hidden_size == 4096


@pytest.mark.smoke
def test_config_from_generator_with_text_config_hidden_size(
    mock_vllm_generator, model_config_text_config
):
    """Test config extraction with text_config.hidden_size attribute."""
    config = create_config(mock_vllm_generator)
    assert config.model.hidden_size == 2048


@pytest.mark.smoke
def test_config_from_generator_extracts_all_settings(
    mock_vllm_generator, model_config_direct
):
    """Test config extracts all settings from generator."""
    config = create_config(mock_vllm_generator)

    assert config.model.target_model_path == mock_vllm_generator.model_path
    assert config.hidden_states.layer_ids == mock_vllm_generator.layer_ids
    assert config.model.tensor_parallel_size == mock_vllm_generator.tensor_parallel_size
    assert (
        config.model.max_model_len
        == mock_vllm_generator.vllm_config.model_config.max_model_len
    )


@pytest.mark.smoke
def test_config_tracks_reproducibility_metadata(
    mock_vllm_generator, model_config_direct
):
    """Test config captures reproducibility metadata."""
    config = create_config(mock_vllm_generator)

    assert config.reproducibility.command
    assert config.reproducibility.package_versions
    assert config.reproducibility.gpu


@pytest.mark.sanity
def test_config_from_generator_fails_with_missing_hidden_size(
    mock_vllm_generator, model_config_missing
):
    """Test config generation fails with helpful error when hidden_size not found."""
    with pytest.raises(ValueError, match="Could not determine hidden size"):
        create_config(mock_vllm_generator)
