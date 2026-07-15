from pathlib import Path

import pytest
from datasets import Dataset as HFDataset

from scripts.prepare_data import assert_safe_to_overwrite, parse_args
from speculators.data_generation import preprocessing as preprocessing_module
from speculators.data_generation.preprocessing import load_and_preprocess_dataset


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


class _FakeProcessor:
    """Minimal processor stub that passes the chat-template precondition."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, *args, **kwargs):
        return ""


def _patch_empty_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make load_and_preprocess_dataset produce an empty dataset without GPU/network."""
    empty = HFDataset.from_dict({"input_ids": [], "loss_mask": [], "seq_len": []})
    monkeypatch.setattr(
        preprocessing_module, "load_processor", lambda *a, **k: _FakeProcessor()
    )
    monkeypatch.setattr(
        preprocessing_module,
        "load_raw_dataset",
        lambda _path: (HFDataset.from_dict({"conversations": []}), None),
    )
    monkeypatch.setattr(
        preprocessing_module, "build_eagle3_dataset", lambda *a, **k: empty
    )
    monkeypatch.setattr(
        preprocessing_module, "save_token_frequency_distribution", lambda **k: None
    )


def test_load_and_preprocess_raises_on_empty_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_empty_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="No samples remain"):
        load_and_preprocess_dataset(
            "target-model",
            ["sharegpt"],
            seq_length=8,
            token_freq_path=tmp_path / "token_freq.pt",
        )


def test_load_and_preprocess_allows_empty_output_with_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _patch_empty_pipeline(monkeypatch)
    dataset, processor = load_and_preprocess_dataset(
        "target-model",
        ["sharegpt"],
        seq_length=8,
        token_freq_path=tmp_path / "token_freq.pt",
        allow_empty_output=True,
    )

    assert len(dataset) == 0
    assert isinstance(processor, _FakeProcessor)
