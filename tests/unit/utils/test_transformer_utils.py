"""
Unit tests for the transformer_utils module in the Speculators library.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

from speculators.utils import transformer_utils

# ===== Fixtures =====


@pytest.fixture
def mock_pretrained_config():
    """Mock PretrainedConfig for testing."""
    config = MagicMock(spec=PretrainedConfig)
    config.name_or_path = "test/model"
    config.to_dict.return_value = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "vocab_size": 50000,
        "model_type": "test_model",
    }
    return config


@pytest.fixture
def mock_pretrained_model():
    """Mock PreTrainedModel for testing."""
    model = MagicMock(spec=PreTrainedModel)
    model.config = MagicMock(spec=PretrainedConfig)
    model.config.to_dict.return_value = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "vocab_size": 50000,
        "model_type": "test_model",
    }
    model.state_dict.return_value = {
        "embedding.weight": torch.randn(50000, 768),
        "layer.0.weight": torch.randn(768, 768),
    }
    return model


@pytest.fixture
def mock_nn_module():
    """Mock nn.Module for testing."""
    module = MagicMock(spec=torch.nn.Module)
    module.state_dict.return_value = {
        "weight": torch.randn(10, 5),
        "bias": torch.randn(10),
    }
    return module


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory with mock checkpoint files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create config.json
        config_data = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "vocab_size": 50000,
            "model_type": "test_model",
        }
        config_file = checkpoint_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Create weight files
        weight_file = checkpoint_path / "pytorch_model.bin"
        torch.save({"weight": torch.randn(10, 5)}, weight_file)

        safetensors_file = checkpoint_path / "model.safetensors"
        safetensors_file.touch()  # Mock file for existence checks

        yield checkpoint_path


@pytest.fixture
def temp_index_checkpoint_dir():
    """Create a temporary directory with indexed checkpoint files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create config.json
        config_data = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "vocab_size": 50000,
            "model_type": "test_model",
        }
        config_file = checkpoint_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Create index file
        index_data = {
            "weight_map": {
                "embedding.weight": "pytorch_model-00001-of-00002.bin",
                "layer.0.weight": "pytorch_model-00002-of-00002.bin",
            }
        }
        index_file = checkpoint_path / "pytorch_model.bin.index.json"
        index_file.write_text(json.dumps(index_data))

        # Create weight files referenced in index
        weight_file_1 = checkpoint_path / "pytorch_model-00001-of-00002.bin"
        torch.save({"embedding.weight": torch.randn(50000, 768)}, weight_file_1)

        weight_file_2 = checkpoint_path / "pytorch_model-00002-of-00002.bin"
        torch.save({"layer.0.weight": torch.randn(768, 768)}, weight_file_2)

        yield checkpoint_path


# ===== download_model_checkpoint_from_hub Tests =====


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.snapshot_download")
def test_download_model_checkpoint_from_hub_success(mock_snapshot_download):
    """Test successful download of model checkpoint from HuggingFace Hub."""
    mock_snapshot_download.return_value = "/path/to/downloaded/model"

    result = transformer_utils.download_model_checkpoint_from_hub("test/model")

    assert result == Path("/path/to/downloaded/model")
    mock_snapshot_download.assert_called_once_with(
        repo_id="test/model",
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        revision=None,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
    )


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.snapshot_download")
def test_download_model_checkpoint_from_hub_with_parameters(mock_snapshot_download):
    """Test download with various parameters."""
    mock_snapshot_download.return_value = "/path/to/downloaded/model"

    result = transformer_utils.download_model_checkpoint_from_hub(
        "test/model",
        cache_dir="/cache",
        force_download=True,
        local_files_only=True,
        token="test_token",
        revision="v1.0",
        custom_param="custom_value",
    )

    assert result == Path("/path/to/downloaded/model")
    mock_snapshot_download.assert_called_once_with(
        repo_id="test/model",
        cache_dir="/cache",
        force_download=True,
        local_files_only=True,
        token="test_token",
        revision="v1.0",
        custom_param="custom_value",
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
    )


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.snapshot_download")
def test_download_model_checkpoint_from_hub_failure(mock_snapshot_download):
    """Test handling of download failure."""
    mock_snapshot_download.side_effect = Exception("Download failed")

    with pytest.raises(FileNotFoundError) as exc_info:
        transformer_utils.download_model_checkpoint_from_hub("test/model")

    assert "Checkpoint not found: test/model" in str(exc_info.value)


# ===== check_download_model_checkpoint Tests =====


@pytest.mark.smoke
def test_check_download_model_checkpoint_with_pretrained_model(mock_pretrained_model):
    """Test with PreTrainedModel instance."""
    result = transformer_utils.check_download_model_checkpoint(mock_pretrained_model)

    assert result is mock_pretrained_model


@pytest.mark.smoke
def test_check_download_model_checkpoint_with_nn_module(mock_nn_module):
    """Test with nn.Module instance."""
    result = transformer_utils.check_download_model_checkpoint(mock_nn_module)

    assert result is mock_nn_module


@pytest.mark.smoke
def test_check_download_model_checkpoint_with_local_path(temp_checkpoint_dir):
    """Test with existing local checkpoint directory."""
    result = transformer_utils.check_download_model_checkpoint(temp_checkpoint_dir)

    assert result == temp_checkpoint_dir.resolve()


@pytest.mark.smoke
def test_check_download_model_checkpoint_invalid():
    """Test with invalid input type."""
    with pytest.raises(TypeError) as exc_info:
        transformer_utils.check_download_model_checkpoint(123)  # type: ignore[arg-type]

    assert "Expected model to be a string or Path" in str(exc_info.value)

    with tempfile.NamedTemporaryFile() as temp_file:
        with pytest.raises(ValueError) as exc_info:  # type: ignore[assignment]
            transformer_utils.check_download_model_checkpoint(temp_file.name)

        assert "Expected a directory for checkpoint" in str(exc_info.value)


@pytest.mark.sanity
@patch("speculators.utils.transformer_utils.download_model_checkpoint_from_hub")
def test_check_download_model_checkpoint_download_from_hub(mock_download):
    """Test download from hub when local path doesn't exist."""
    mock_download.return_value = Path("/downloaded/model")

    result = transformer_utils.check_download_model_checkpoint(
        "nonexistent/path",
        cache_dir="/cache",
        force_download=True,
        token="test_token",
    )

    assert result == Path("/downloaded/model")
    mock_download.assert_called_once_with(
        model_id="nonexistent/path",
        cache_dir="/cache",
        force_download=True,
        local_files_only=False,
        token="test_token",
        revision=None,
    )


# ===== check_download_model_config Tests =====


@pytest.mark.smoke
def test_check_download_model_config_with_pretrained_config(mock_pretrained_config):
    """Test with PretrainedConfig instance."""
    result = transformer_utils.check_download_model_config(mock_pretrained_config)

    assert result is mock_pretrained_config


@pytest.mark.smoke
def test_check_download_model_config_with_pretrained_model(mock_pretrained_model):
    """Test with PreTrainedModel instance."""
    result = transformer_utils.check_download_model_config(mock_pretrained_model)

    assert result is mock_pretrained_model.config


@pytest.mark.smoke
def test_check_download_model_config_with_dict():
    """Test with dictionary config."""
    config_dict = {"model_type": "test", "hidden_size": 768}
    result = transformer_utils.check_download_model_config(config_dict)

    assert result is config_dict


@pytest.mark.smoke
def test_check_download_model_config_with_local_file(temp_checkpoint_dir):
    """Test with existing local config file."""
    config_path = temp_checkpoint_dir / "config.json"
    result = transformer_utils.check_download_model_config(config_path)

    assert result == config_path.resolve()


@pytest.mark.smoke
def test_check_download_model_config_with_local_dir(temp_checkpoint_dir):
    """Test with existing local checkpoint directory."""
    result = transformer_utils.check_download_model_config(temp_checkpoint_dir)

    assert result == (temp_checkpoint_dir / "config.json").resolve()


@pytest.mark.smoke
def test_check_download_model_config_invalid():
    """Test with invalid input type."""
    with pytest.raises(TypeError) as exc_info:
        transformer_utils.check_download_model_config(123)  # type: ignore[arg-type]

    assert "Expected config to be a string, Path, or PreTrainedModel" in str(
        exc_info.value
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        missing_config_path = Path(temp_dir) / "missing_dir"

        with pytest.raises(OSError) as exc_info:  # type: ignore[assignment]
            transformer_utils.check_download_model_config(missing_config_path)

        assert "Can't load the configuration" in str(exc_info.value)


@pytest.mark.sanity
@patch("speculators.utils.transformer_utils.AutoConfig")
def test_check_download_model_config_download_from_hub(mock_auto_config):
    """Test download from hub when local path doesn't exist."""
    mock_config = MagicMock(spec=PretrainedConfig)
    mock_auto_config.from_pretrained.return_value = mock_config

    result = transformer_utils.check_download_model_config(
        "nonexistent/path",
        cache_dir="/cache",
        force_download=True,
        token="test_token",
    )

    assert result is mock_config
    mock_auto_config.from_pretrained.assert_called_once_with(
        "nonexistent/path",
        cache_dir="/cache",
        force_download=True,
        local_files_only=False,
        token="test_token",
        revision=None,
    )


# ===== load_model_config Tests =====


@pytest.mark.smoke
def test_load_model_config_with_pretrained_config(mock_pretrained_config):
    """Test with PretrainedConfig instance."""
    result = transformer_utils.load_model_config(mock_pretrained_config)

    assert result is mock_pretrained_config


@pytest.mark.smoke
def test_load_model_config_with_pretrained_model(mock_pretrained_model):
    """Test with PreTrainedModel instance."""
    result = transformer_utils.load_model_config(mock_pretrained_model)

    assert result is mock_pretrained_model.config


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.AutoConfig")
def test_load_model_config_from_path(mock_auto_config):
    """Test loading config from path."""
    mock_config = MagicMock(spec=PretrainedConfig)
    mock_auto_config.from_pretrained.return_value = mock_config

    result = transformer_utils.load_model_config(
        "test/model",
        cache_dir="/cache",
        force_download=True,
        token="test_token",
    )

    assert result is mock_config
    mock_auto_config.from_pretrained.assert_called_once_with(
        "test/model",
        cache_dir="/cache",
        force_download=True,
        local_files_only=False,
        token="test_token",
        revision=None,
    )


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.AutoConfig")
def test_load_model_config_invalid(mock_auto_config):
    """Test with invalid input type."""
    with pytest.raises(TypeError) as exc_info:
        transformer_utils.load_model_config(123)  # type: ignore[arg-type]

    assert "Expected model to be a string, Path, or PreTrainedModel" in str(
        exc_info.value
    )

    mock_auto_config.from_pretrained.side_effect = ValueError("Config not found")

    with pytest.raises(FileNotFoundError) as exc_info:  # type: ignore[assignment]
        transformer_utils.load_model_config("test/model")

    assert "Config not found for model: test/model" in str(exc_info.value)


# ===== load_model_checkpoint_config_dict Tests =====


@pytest.mark.smoke
def test_load_model_checkpoint_config_dict_with_dict():
    """Test with dictionary input."""
    config_dict = {"model_type": "test", "hidden_size": 768}
    result = transformer_utils.load_model_checkpoint_config_dict(config_dict)

    assert result is config_dict


@pytest.mark.smoke
def test_load_model_checkpoint_config_dict_with_pretrained_model(mock_pretrained_model):
    """Test with PreTrainedModel instance."""
    result = transformer_utils.load_model_checkpoint_config_dict(mock_pretrained_model)

    assert result == mock_pretrained_model.config.to_dict.return_value
    mock_pretrained_model.config.to_dict.assert_called_once()


@pytest.mark.smoke
def test_load_model_checkpoint_config_dict_with_pretrained_config(
    mock_pretrained_config,
):
    """Test with PretrainedConfig instance."""
    result = transformer_utils.load_model_checkpoint_config_dict(mock_pretrained_config)

    assert result == mock_pretrained_config.to_dict.return_value
    mock_pretrained_config.to_dict.assert_called_once()


@pytest.mark.smoke
def test_load_model_checkpoint_config_dict_with_file(temp_checkpoint_dir):
    """Test with config file path."""
    config_path = temp_checkpoint_dir / "config.json"
    result = transformer_utils.load_model_checkpoint_config_dict(config_path)

    expected_config = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "vocab_size": 50000,
        "model_type": "test_model",
    }
    assert result == expected_config


@pytest.mark.smoke
def test_load_model_checkpoint_config_dict_with_dir(temp_checkpoint_dir):
    """Test with checkpoint directory."""
    result = transformer_utils.load_model_checkpoint_config_dict(temp_checkpoint_dir)

    expected_config = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "vocab_size": 50000,
        "model_type": "test_model",
    }
    assert result == expected_config


@pytest.mark.smoke
def test_load_model_checkpoint_config_dict_invalid():
    """Test with invalid input type."""
    with pytest.raises(TypeError) as exc_info:
        transformer_utils.load_model_checkpoint_config_dict(123)  # type: ignore[arg-type]

    assert (
        "Expected config to be a string, Path, PreTrainedModel, or PretrainedConfig"
        in str(exc_info.value)
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        missing_config_path = Path(temp_dir) / "config.json"

        with pytest.raises(FileNotFoundError) as exc_info:  # type: ignore[assignment]
            transformer_utils.load_model_checkpoint_config_dict(missing_config_path)

        assert "No config.json found" in str(exc_info.value)


# ===== load_model_checkpoint_index_weight_files Tests =====


@pytest.mark.smoke
def test_load_model_checkpoint_index_weight_files_with_directory(
    temp_index_checkpoint_dir,
):
    """Test with directory containing index files."""
    result = transformer_utils.load_model_checkpoint_index_weight_files(
        temp_index_checkpoint_dir
    )

    assert len(result) == 2
    assert all(isinstance(f, Path) for f in result)
    assert all(f.exists() for f in result)


@pytest.mark.smoke
def test_load_model_checkpoint_index_weight_files_no_index_files():
    """Test with directory containing no index files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = transformer_utils.load_model_checkpoint_index_weight_files(temp_dir)
        assert result == []


@pytest.mark.smoke
def test_load_model_checkpoint_index_weight_files_invalid():
    """Test with invalid input type."""
    with pytest.raises(TypeError) as exc_info:
        transformer_utils.load_model_checkpoint_index_weight_files(123)  # type: ignore[arg-type]

    assert "Expected path to be a string or Path" in str(exc_info.value)

    with pytest.raises(FileNotFoundError) as exc_info:  # type: ignore[assignment]
        transformer_utils.load_model_checkpoint_index_weight_files("/nonexistent/path")

    assert "Model checkpoint path does not exist" in str(exc_info.value)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create invalid index file
        invalid_index_data = {"metadata": {"total_size": 1000}}
        index_file = checkpoint_path / "pytorch_model.bin.index.json"
        index_file.write_text(json.dumps(invalid_index_data))

        # When processing the directory, this should raise a ValueError
        with pytest.raises(ValueError) as exc_info:  # type: ignore[assignment]
            transformer_utils.load_model_checkpoint_index_weight_files(checkpoint_path)

        assert "does not contain a weight_map" in str(exc_info.value)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create index file with non-existent weight file
        index_data = {
            "weight_map": {
                "embedding.weight": "missing_file.bin",
            }
        }
        index_file = checkpoint_path / "pytorch_model.bin.index.json"
        index_file.write_text(json.dumps(index_data))

        # When processing the directory, this should raise a FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:  # type: ignore[assignment]
            transformer_utils.load_model_checkpoint_index_weight_files(checkpoint_path)

        assert "Weight file for" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)


# ===== load_model_checkpoint_weight_files Tests =====


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_with_bin_file(temp_checkpoint_dir):
    """Test with single .bin file."""
    bin_file = temp_checkpoint_dir / "pytorch_model.bin"
    result = transformer_utils.load_model_checkpoint_weight_files(bin_file)

    assert len(result) == 1
    assert result[0] == bin_file


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_with_safetensors_file(temp_checkpoint_dir):
    """Test with single .safetensors file."""
    safetensors_file = temp_checkpoint_dir / "model.safetensors"
    result = transformer_utils.load_model_checkpoint_weight_files(safetensors_file)

    assert len(result) == 1
    assert result[0] == safetensors_file


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_with_directory_bin_only():
    """Test with directory containing only .bin files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create only bin files (no safetensors)
        bin_file = checkpoint_path / "pytorch_model.bin"
        torch.save({"weight": torch.randn(10, 5)}, bin_file)

        result = transformer_utils.load_model_checkpoint_weight_files(checkpoint_path)

        assert len(result) == 1
        assert result[0].suffix == ".bin"


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_with_directory_bin(temp_checkpoint_dir):
    """Test with directory containing .bin files."""
    result = transformer_utils.load_model_checkpoint_weight_files(temp_checkpoint_dir)

    # Should return .safetensors files first as they are preferred
    assert len(result) == 1
    assert result[0].suffix == ".safetensors"


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_with_directory_safetensors():
    """Test with directory containing .safetensors files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create only safetensors files
        safetensors_file = checkpoint_path / "model.safetensors"
        safetensors_file.touch()

        result = transformer_utils.load_model_checkpoint_weight_files(checkpoint_path)

        assert len(result) == 1
        assert result[0].suffix == ".safetensors"


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_with_index_files(temp_index_checkpoint_dir):
    """Test with directory containing index files."""
    result = transformer_utils.load_model_checkpoint_weight_files(
        temp_index_checkpoint_dir
    )

    assert len(result) == 2
    assert all(f.suffix == ".bin" for f in result)


@pytest.mark.smoke
def test_load_model_checkpoint_weight_files_invalid():
    """Test with invalid input type."""
    with pytest.raises(TypeError) as exc_info:
        transformer_utils.load_model_checkpoint_weight_files(123)  # type: ignore[arg-type]

    assert "Expected path to be a string or Path" in str(exc_info.value)

    with pytest.raises(FileNotFoundError) as exc_info:  # type: ignore[assignment]
        transformer_utils.load_model_checkpoint_weight_files("/nonexistent/path")

    assert "Model checkpoint path does not exist" in str(exc_info.value)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create a non-weight file
        other_file = checkpoint_path / "README.md"
        other_file.write_text("This is a readme")

        with pytest.raises(FileNotFoundError) as exc_info:  # type: ignore[assignment]
            transformer_utils.load_model_checkpoint_weight_files(checkpoint_path)

        assert "No valid weight files found" in str(exc_info.value)


# ===== load_model_checkpoint_state_dict Tests =====


@pytest.mark.smoke
def test_load_model_checkpoint_state_dict_with_pretrained_model(mock_pretrained_model):
    """Test with PreTrainedModel instance."""
    result = transformer_utils.load_model_checkpoint_state_dict(mock_pretrained_model)

    assert result == mock_pretrained_model.state_dict.return_value
    mock_pretrained_model.state_dict.assert_called_once()


@pytest.mark.smoke
def test_load_model_checkpoint_state_dict_with_nn_module(mock_nn_module):
    """Test with nn.Module instance."""
    result = transformer_utils.load_model_checkpoint_state_dict(mock_nn_module)

    assert result == mock_nn_module.state_dict.return_value
    mock_nn_module.state_dict.assert_called_once()


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.torch.load")
def test_load_model_checkpoint_state_dict_with_bin_file(
    mock_torch_load, temp_checkpoint_dir
):
    """Test with .bin file."""
    bin_file = temp_checkpoint_dir / "pytorch_model.bin"
    mock_torch_load.return_value = {
        "embedding.weight": torch.randn(50000, 768),
        "layer.0.weight": torch.randn(768, 768),
    }

    result = transformer_utils.load_model_checkpoint_state_dict(bin_file)

    assert len(result) == 2
    assert "embedding.weight" in result
    assert "layer.0.weight" in result
    mock_torch_load.assert_called_once_with(bin_file, map_location="cpu")


@pytest.mark.smoke
@patch("speculators.utils.transformer_utils.safe_open")
def test_load_model_checkpoint_state_dict_with_safetensors_file(
    mock_safe_open, temp_checkpoint_dir
):
    """Test with .safetensors file."""
    safetensors_file = temp_checkpoint_dir / "model.safetensors"

    # Mock the safe_open context manager
    mock_safetensors_file = MagicMock()
    mock_safetensors_file.keys.return_value = ["embedding.weight", "layer.0.weight"]
    mock_safetensors_file.get_tensor.side_effect = lambda key: torch.randn(768, 768)
    mock_safe_open.return_value.__enter__.return_value = mock_safetensors_file

    result = transformer_utils.load_model_checkpoint_state_dict(safetensors_file)

    assert len(result) == 2
    assert "embedding.weight" in result
    assert "layer.0.weight" in result
    mock_safe_open.assert_called_once_with(
        safetensors_file, framework="pt", device="cpu"
    )


@pytest.mark.sanity
@patch("speculators.utils.transformer_utils.torch.load")
def test_load_model_checkpoint_state_dict_with_index_files(
    mock_torch_load, temp_index_checkpoint_dir
):
    """Test with directory containing index files."""
    mock_torch_load.side_effect = [
        {"embedding.weight": torch.randn(50000, 768)},
        {"layer.0.weight": torch.randn(768, 768)},
    ]

    result = transformer_utils.load_model_checkpoint_state_dict(
        temp_index_checkpoint_dir
    )

    assert len(result) == 2
    assert "embedding.weight" in result
    assert "layer.0.weight" in result
    assert mock_torch_load.call_count == 2


@pytest.mark.smoke
def test_load_model_checkpoint_state_dict_unsupported_file_type():
    """Test with unsupported file type."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create an unsupported file type
        unsupported_file = checkpoint_path / "model.txt"
        unsupported_file.write_text("This is not a weight file")

        with pytest.raises(FileNotFoundError) as exc_info:
            transformer_utils.load_model_checkpoint_state_dict(unsupported_file)

        assert "No valid weight files found" in str(exc_info.value)


@pytest.mark.sanity
@patch("speculators.utils.transformer_utils.torch.load")
@patch("speculators.utils.transformer_utils.safe_open")
def test_load_model_checkpoint_state_dict_mixed_file_types(
    mock_safe_open, mock_torch_load
):
    """Test with directory containing both .bin and .safetensors files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create both file types
        bin_file = checkpoint_path / "pytorch_model.bin"
        bin_file.touch()
        safetensors_file = checkpoint_path / "model.safetensors"
        safetensors_file.touch()

        # Mock torch.load
        mock_torch_load.return_value = {"bin_weight": torch.randn(10, 10)}

        # Mock safe_open
        mock_safetensors_file = MagicMock()
        mock_safetensors_file.keys.return_value = ["safetensors_weight"]
        mock_safetensors_file.get_tensor.return_value = torch.randn(20, 20)
        mock_safe_open.return_value.__enter__.return_value = mock_safetensors_file

        result = transformer_utils.load_model_checkpoint_state_dict(checkpoint_path)

        # Should prefer safetensors files
        assert len(result) == 1
        assert "safetensors_weight" in result
        mock_safe_open.assert_called_once()
        mock_torch_load.assert_not_called()
