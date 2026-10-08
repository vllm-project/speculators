import json
from pathlib import Path

import pytest
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from transformers import AutoProcessor, AutoTokenizer
from typer.testing import CliRunner

from speculators.cli import app
from speculators.cli.prepare_data import assert_safe_to_overwrite
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


@pytest.fixture
def no_local_model(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Data preparation must not load a local model or tokenizer")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", fail)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail)


@pytest.mark.parametrize("allow_empty", [False, True])
def test_load_and_preprocess_empty_output(tmp_path, no_local_model, allow_empty):
    source = tmp_path / "empty_supervision.jsonl"
    source.write_text(json.dumps({"input_ids": [1, 2], "loss_mask": [0, 0]}) + "\n")
    kwargs = {
        "seq_length": 8,
        "build_dataset_num_proc": 1,
        "allow_empty_output": allow_empty,
    }
    if allow_empty:
        dataset = load_and_preprocess_dataset([str(source)], **kwargs)
        assert len(dataset) == 0
    else:
        with pytest.raises(ValueError, match="No samples remain"):
            load_and_preprocess_dataset([str(source)], **kwargs)


def test_prepare_token_rows_without_model_or_server(
    tmp_path, monkeypatch, no_local_model
):
    def fail(*args, **kwargs):
        pytest.fail("Prepared rows must not call the render endpoint")

    monkeypatch.setattr(preprocessing_module, "render_conversation", fail)
    source = tmp_path / "tokens.jsonl"
    source.write_text(
        json.dumps({"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]}) + "\n"
    )
    output = tmp_path / "prepared"
    result = CliRunner(env={"COLUMNS": "200"}).invoke(
        app,
        [
            "prepare-data",
            "--data",
            str(source),
            "--output",
            str(output),
            "--seq-length",
            "3",
            "--num-preprocessing-workers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    dataset = load_from_disk(str(output))
    assert dataset[0]["input_ids"].tolist() == [1, 2, 3]
    assert dataset[0]["loss_mask"].tolist() == [0, 1, 1]


def test_huggingface_jsonl_uri_downloads_dataset_file(monkeypatch: pytest.MonkeyPatch):
    raw = HFDataset.from_dict({"input_ids": [[1, 2]], "loss_mask": [[0, 1]]})
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return "/tmp/regenerated.jsonl"

    def fake_load_dataset(*args, **kwargs):
        calls["load_dataset"] = (args, kwargs)
        return raw

    monkeypatch.setattr(preprocessing_module, "hf_hub_download", fake_download)
    monkeypatch.setattr(preprocessing_module, "load_dataset", fake_load_dataset)

    loaded, normalize_fn = preprocessing_module.load_raw_dataset(
        "hf://datasets/example-org/regenerated-responses/qwen3.jsonl"
    )

    assert loaded is raw
    assert normalize_fn is None
    assert calls["repo_id"] == "example-org/regenerated-responses"
    assert calls["filename"] == "qwen3.jsonl"
    assert calls["repo_type"] == "dataset"
    assert calls["load_dataset"][0] == ("json",)
