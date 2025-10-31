"""Unit tests for data generation config generator.

Tests cover:
- Package version detection
- GPU info detection
- Cache key generation
- Example data generation
- Config dataclass creation
- Full config generation
- Generator extraction
"""

import json
from unittest.mock import Mock, patch

import pytest
import torch

from speculators.data_generation.config_generator import (
    DataGenerationConfig,
    ExampleData,
    FormatConfig,
    PackageVersions,
    ReproducibilityInfo,
    _get_gpu_info,
    _get_hidden_size_from_model,
    extract_config_from_generator,
    generate_config,
    generate_example_data,
)


@pytest.fixture
def mock_hidden_size():
    """Mock the _get_hidden_size_from_model function."""
    with patch(
        "speculators.data_generation.config_generator._get_hidden_size_from_model"
    ) as mock:
        mock.return_value = 4096
        yield mock


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=True: (
        [1, 2, 3, 4, 5] if "quick" in text else [10, 11, 12]
    )
    tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: (
        "decoded text" if len(ids) == 5 else "output text"
    )
    return tokenizer


@pytest.fixture
def mock_vllm_generator():
    """Create a mock VllmHiddenStatesGenerator for testing."""
    generator = Mock()
    generator.model_path = "test-model/test-7b"
    generator.layer_ids = [2, 14, 24, 31]
    generator.tensor_parallel_size = 1

    generator.tokenizer = Mock()
    generator.tokenizer.encode.side_effect = [[1, 2, 3, 4, 5], [10, 11, 12]]
    generator.tokenizer.decode.side_effect = ["test prompt", "test output"]

    generator.vllm_config = Mock()
    generator.vllm_config.model_config = Mock()
    generator.vllm_config.model_config.max_model_len = 2048
    generator.vllm_config.cache_config = Mock()
    generator.vllm_config.cache_config.gpu_memory_utilization = 0.8

    generator.llm = Mock()
    generator.sampling_params = Mock()

    return generator


def test_package_versions_dataclass():
    """Test PackageVersions dataclass creation."""
    versions = PackageVersions(
        torch="2.0.0", vllm="0.6.0", transformers="4.40.0", speculators="0.1.0"
    )

    assert versions.torch == "2.0.0"
    assert versions.vllm == "0.6.0"
    assert versions.transformers == "4.40.0"
    assert versions.speculators == "0.1.0"


def test_package_versions_from_environment():
    """Test detecting package versions from environment."""
    versions = PackageVersions.from_environment()

    assert versions.torch is not None
    assert versions.transformers is not None
    assert versions.vllm is not None or versions.vllm == "unknown"
    assert versions.torch == torch.__version__


def test_reproducibility_info_creation():
    """Test ReproducibilityInfo creation."""
    versions = PackageVersions(
        torch="2.0.0", vllm="0.6.0", transformers="4.40.0", speculators="0.1.0"
    )

    info = ReproducibilityInfo.create(
        command="python script.py --arg value",
        cache_key="abc123",
        package_versions=versions,
    )

    assert info.command == "python script.py --arg value"
    assert info.cache_key == "abc123"
    assert info.gpu is not None
    assert info.packages["torch"] == "2.0.0"
    assert info.packages["vllm"] == "0.6.0"


def test_format_config_default():
    """Test FormatConfig creation with defaults."""
    config = FormatConfig.create_default(num_layers=4, hidden_size=4096)

    assert config.file_pattern == "data_{idx}.pt"
    assert "input_ids" in config.schema
    assert "hidden_states" in config.schema
    assert "loss_mask" in config.schema
    assert config.schema["input_ids"]["dtype"] == "torch.long"
    assert config.schema["hidden_states"]["num_tensors"] == 4
    assert "4096" in config.schema["hidden_states"]["shape"]


def test_get_gpu_info():
    """Test GPU info detection."""
    info = _get_gpu_info()

    assert isinstance(info, str)
    assert len(info) > 0
    assert any(
        keyword in info.lower() for keyword in ["gpu", "cuda", "cpu", "nvidia", "amd"]
    )


def test_package_versions_speculators_version():
    """Test that PackageVersions can detect speculators version."""
    versions = PackageVersions.from_environment()

    assert versions.speculators is None or isinstance(versions.speculators, str)


def test_get_hidden_size_from_model():
    """Test extracting hidden size from model config."""
    with patch("transformers.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value = Mock(hidden_size=4096)

        hidden_size = _get_hidden_size_from_model("test-model")

        assert hidden_size == 4096
        mock_config.assert_called_once_with("test-model", trust_remote_code=True)


def test_get_hidden_size_from_model_text_config():
    """Test extracting hidden size from text_config."""
    with patch("transformers.AutoConfig.from_pretrained") as mock_config:
        config = Mock(spec=["text_config"])
        config.text_config = Mock(hidden_size=2048)
        mock_config.return_value = config

        hidden_size = _get_hidden_size_from_model("test-model")

        assert hidden_size == 2048


def test_get_hidden_size_from_model_error():
    """Test error handling when hidden size not found."""
    with patch("transformers.AutoConfig.from_pretrained") as mock_config:
        config = Mock(spec=[])
        mock_config.return_value = config

        with pytest.raises(ValueError, match="Could not"):
            _get_hidden_size_from_model("test-model")


def test_generate_example_data(mock_tokenizer):
    """Test generating example data."""
    prompt_token_ids = [1, 2, 3, 4, 5]
    output_token_ids = [10, 11, 12]

    example = generate_example_data(mock_tokenizer, prompt_token_ids, output_token_ids)

    assert isinstance(example, ExampleData)
    assert example.prompt_token_ids == prompt_token_ids
    assert example.output_token_ids == output_token_ids
    assert isinstance(example.prompt_str, str)
    assert isinstance(example.output_str, str)
    mock_tokenizer.decode.assert_called()


def test_generate_config_basic(mock_hidden_size):
    """Test basic config generation."""
    with patch(
        "speculators.data_generation.config_generator.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer_load:
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [[1, 2, 3, 4, 5], [10, 11, 12]]
        mock_tokenizer.decode.side_effect = ["prompt text", "output text"]
        mock_tokenizer_load.return_value = mock_tokenizer

        config = generate_config(
            target_model_path="test-model",
            train_data_path="sharegpt",
            chat_template="llama3",
            seq_length=2048,
            layer_ids=[2, 14, 24, 31],
            tensor_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            batch_size=8,
            cache_dir="./cache",
            num_samples=1000,
            example_prompt_token_ids=[1, 2, 3, 4, 5],
            example_output_token_ids=[10, 11, 12],
            seed=0,
        )

        assert isinstance(config, DataGenerationConfig)
        assert config.version == "2.0"
        assert config.model["target_model_path"] == "test-model"
        assert config.data["train_data_path"] == "sharegpt"
        assert config.hidden_states["layer_ids"] == [2, 14, 24, 31]
        assert len(config.example_prompt_token_ids) > 0
        assert len(config.example_output_token_ids) > 0


def test_generate_config_with_tokenizer(mock_hidden_size):
    """Test config generation with pre-loaded tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.side_effect = [[1, 2, 3, 4, 5], [10, 11, 12]]
    mock_tokenizer.decode.side_effect = ["prompt", "output"]

    generate_config(
        target_model_path="test-model",
        train_data_path="sharegpt",
        chat_template="llama3",
        seq_length=2048,
        layer_ids=[2, 14, 24, 31],
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        batch_size=8,
        cache_dir="./cache",
        num_samples=1000,
        example_prompt_token_ids=[1, 2, 3, 4, 5],
        example_output_token_ids=[10, 11, 12],
        tokenizer=mock_tokenizer,
    )

    mock_tokenizer.decode.assert_called()


def test_config_json_serializable(mock_hidden_size):
    """Test that config can be serialized to JSON."""
    with patch(
        "speculators.data_generation.config_generator.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer_load:
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [[1, 2, 3], [4, 5, 6]]
        mock_tokenizer.decode.side_effect = ["prompt", "output"]
        mock_tokenizer_load.return_value = mock_tokenizer

        config = generate_config(
            target_model_path="test-model",
            train_data_path="sharegpt",
            chat_template="llama3",
            seq_length=2048,
            layer_ids=[2, 14, 24, 31],
            tensor_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            batch_size=8,
            cache_dir="./cache",
            num_samples=1000,
            example_prompt_token_ids=[1, 2, 3],
            example_output_token_ids=[4, 5, 6],
        )

        json_str = json.dumps(config.to_dict(), indent=2)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["version"] == "2.0"


def test_extract_config_from_generator(mock_vllm_generator, mock_hidden_size):
    """Test extracting config from VllmHiddenStatesGenerator."""
    mock_output = Mock()
    mock_output.outputs = [Mock(token_ids=[10, 11, 12])]
    mock_vllm_generator.llm.generate.return_value = [mock_output]

    config = extract_config_from_generator(
        generator=mock_vllm_generator,
        train_data_path="sharegpt",
        chat_template="llama3",
        seq_length=2048,
        batch_size=8,
        cache_dir="./cache",
        num_samples=1000,
        seed=0,
    )

    assert config.model["target_model_path"] == mock_vllm_generator.model_path
    assert (
        config.model["tensor_parallel_size"] == mock_vllm_generator.tensor_parallel_size
    )
    assert config.hidden_states["layer_ids"] == mock_vllm_generator.layer_ids
    mock_vllm_generator.llm.generate.assert_called_once()


def test_extract_config_uses_generator_tokenizer(mock_vllm_generator, mock_hidden_size):
    """Test that extraction reuses generator's tokenizer."""
    mock_output = Mock()
    mock_output.outputs = [Mock(token_ids=[10, 11, 12])]
    mock_vllm_generator.llm.generate.return_value = [mock_output]

    extract_config_from_generator(
        generator=mock_vllm_generator,
        train_data_path="sharegpt",
        chat_template="llama3",
        seq_length=2048,
        batch_size=8,
        cache_dir="./cache",
        num_samples=1000,
    )

    mock_vllm_generator.tokenizer.encode.assert_called()
    mock_vllm_generator.tokenizer.decode.assert_called()


def test_generate_config_with_max_samples_none(mock_hidden_size):
    """Test config generation with max_samples=None."""
    with patch(
        "speculators.data_generation.config_generator.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer_load:
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [[1, 2, 3], [4, 5, 6]]
        mock_tokenizer.decode.side_effect = ["prompt", "output"]
        mock_tokenizer_load.return_value = mock_tokenizer

        config = generate_config(
            target_model_path="test-model",
            train_data_path="sharegpt",
            chat_template="llama3",
            seq_length=2048,
            layer_ids=[2, 14, 24, 31],
            tensor_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            batch_size=8,
            cache_dir="./cache",
            num_samples=1000,
            example_prompt_token_ids=[1, 2, 3],
            example_output_token_ids=[4, 5, 6],
            max_samples=None,
        )

        assert config.data["max_samples"] is None
        assert config.data["num_samples"] == 1000


def test_reproducibility_info_contains_all_fields():
    """Test that reproducibility info has all required fields."""
    versions = PackageVersions(torch="2.0.0", vllm="0.6.0", transformers="4.40.0")

    info = ReproducibilityInfo.create(
        command="test command", cache_key="abc123", package_versions=versions
    )

    assert info.packages["torch"] == "2.0.0"
    assert info.packages["vllm"] == "0.6.0"
    assert info.packages["transformers"] == "4.40.0"


def test_format_config_schema_structure():
    """Test that format config schema has correct structure."""
    config = FormatConfig.create_default(num_layers=4, hidden_size=4096)

    for _field_name, field_schema in config.schema.items():
        assert "dtype" in field_schema
        assert "shape" in field_schema

    assert "num_tensors" in config.schema["hidden_states"]
    assert config.schema["hidden_states"]["num_tensors"] == 4
