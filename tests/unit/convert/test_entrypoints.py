"""Unit tests for the external-checkpoint auto-conversion dispatch."""

from unittest.mock import patch

import pytest

from speculators.convert.entrypoints import maybe_convert_external_checkpoint


class TestMaybeConvertExternalCheckpoint:
    @patch("speculators.convert.entrypoints.PretrainedConfig.get_config_dict")
    def test_speculators_checkpoint_passes_through(self, mock_cfg):
        mock_cfg.return_value = ({"speculators_model_type": "dflash"}, None)
        assert maybe_convert_external_checkpoint("some/speculators-model") == (
            "some/speculators-model"
        )

    @patch("speculators.convert.entrypoints.convert_model")
    @patch("speculators.convert.entrypoints.PretrainedConfig.get_config_dict")
    def test_external_dflash_converts(self, mock_cfg, mock_convert):
        mock_cfg.return_value = ({"dflash_config": {}, "model_type": "qwen3"}, None)
        out = maybe_convert_external_checkpoint(
            "z-lab/Qwen3-8B-DFlash-b16",
            verifier="Qwen/Qwen3-8B",
            output_path="/tmp/out",
            force_download=True,
            local_files_only=True,
            token="token",
            revision="test-revision",
        )
        assert out == "/tmp/out"
        mock_cfg.assert_called_once_with(
            "z-lab/Qwen3-8B-DFlash-b16",
            cache_dir=None,
            force_download=True,
            local_files_only=True,
            token="token",
            revision="test-revision",
        )
        mock_convert.assert_called_once()
        assert mock_convert.call_args.kwargs["algorithm"] == "dflash"
        assert mock_convert.call_args.kwargs["verifier"] == "Qwen/Qwen3-8B"
        assert mock_convert.call_args.kwargs["force_download"] is True
        assert mock_convert.call_args.kwargs["local_files_only"] is True
        assert mock_convert.call_args.kwargs["token"] == "token"
        assert mock_convert.call_args.kwargs["revision"] == "test-revision"

    @patch("speculators.convert.entrypoints.PretrainedConfig.get_config_dict")
    def test_external_without_verifier_raises(self, mock_cfg):
        mock_cfg.return_value = ({"architectures": ["DFlashDraftModel"]}, None)
        with pytest.raises(ValueError, match="requires a verifier"):
            maybe_convert_external_checkpoint("z-lab/Qwen3-8B-DFlash-b16")

    @patch("speculators.convert.entrypoints.PretrainedConfig.get_config_dict")
    def test_unrecognized_format_raises(self, mock_cfg):
        mock_cfg.return_value = ({"model_type": "qwen3"}, None)
        with pytest.raises(NotImplementedError, match="unrecognized external"):
            maybe_convert_external_checkpoint("some/model", verifier="Qwen/Qwen3-8B")
