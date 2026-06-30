from pathlib import Path

import pytest

from scripts.prepare_data import assert_safe_to_overwrite, parse_args


def test_assert_safe_to_overwrite_allows_prepare_data_artifacts(tmp_path: Path):
    output = tmp_path / "data"
    output.mkdir()
    (output / "data-00000-of-00001.arrow").touch()
    (output / "dataset_info.json").touch()
    token_freq_path = output / "token_freq.pt"
    token_freq_path.touch()

    assert_safe_to_overwrite(output, token_freq_path)


def test_assert_safe_to_overwrite_rejects_unknown_files(tmp_path: Path):
    output = tmp_path / "data"
    output.mkdir()
    (output / "data-00000-of-00001.arrow").touch()
    (output / "checkpoints").mkdir()

    with pytest.raises(ValueError, match="would delete files"):
        assert_safe_to_overwrite(output, output / "token_freq.pt")


def test_assert_safe_to_overwrite_honors_custom_token_freq_path(tmp_path: Path):
    output = tmp_path / "data"
    output.mkdir()
    token_freq_path = output / "custom_freq.pt"
    token_freq_path.touch()

    assert_safe_to_overwrite(output, token_freq_path)


def test_parse_args_forwards_allow_empty_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--model",
            "target",
            "--data",
            "sharegpt",
            "--output",
            "out",
            "--allow-empty-output",
        ],
    )

    args = parse_args()

    assert args.allow_empty_output is True


def test_parse_args_allow_empty_output_defaults_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data.py", "--model", "target", "--data", "sharegpt", "--output", "o"],
    )

    args = parse_args()

    assert args.allow_empty_output is False
