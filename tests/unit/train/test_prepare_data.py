import importlib
import logging
from pathlib import Path

import pytest
import typer
from datasets import Dataset as HFDataset

from speculators.cli.prepare_data import assert_safe_to_overwrite, prepare_data
from speculators.data_generation import preprocessing as preprocessing_module
from speculators.data_generation.preprocessing import load_and_preprocess_dataset

# The CLI package binds the command function to this name, which shadows the
# submodule. Patch the module object so the command sees the test double.
_prepare_data_module = importlib.import_module("speculators.cli.prepare_data")


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


def _forbid_dataset_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(*_args, **_kwargs):
        raise AssertionError("prepare-data must not rebuild this output directory")

    monkeypatch.setattr(_prepare_data_module, "load_and_preprocess_dataset", fail_load)


def _write_arrow_output(path: Path, metadata: list[str]) -> None:
    path.mkdir()
    (path / "data-00000-of-00001.arrow").write_text("shard")
    for name in metadata:
        (path / name).write_text("{}")


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        ["dataset_info.json"],
        ["state.json"],
    ],
)
def test_prepare_data_rejects_partial_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    metadata: list[str],
):
    output = tmp_path / "out"
    _write_arrow_output(output, metadata)
    _forbid_dataset_load(monkeypatch)

    with (
        caplog.at_level(logging.ERROR, logger="speculators.cli.prepare_data"),
        pytest.raises(typer.Exit) as exc,
    ):
        prepare_data(
            model="target-model",
            data=["conversations.jsonl"],
            output=str(output),
        )

    assert exc.value.exit_code == 1, (
        f"partial dataset directory exited {exc.value.exit_code}; "
        "an incomplete save must not be treated as finished"
    )
    assert "incomplete" in caplog.text.lower()
    assert "--overwrite" in caplog.text
    assert (output / "data-00000-of-00001.arrow").read_text() == "shard"


def test_prepare_data_skips_complete_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    output = tmp_path / "out"
    _write_arrow_output(output, ["dataset_info.json", "state.json"])
    _forbid_dataset_load(monkeypatch)

    with (
        caplog.at_level(logging.INFO, logger="speculators.cli.prepare_data"),
        pytest.raises(typer.Exit) as exc,
    ):
        prepare_data(
            model="target-model",
            data=["conversations.jsonl"],
            output=str(output),
        )

    assert exc.value.exit_code == 0
    assert "skipping" in caplog.text.lower()
    assert (output / "data-00000-of-00001.arrow").read_text() == "shard"


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        ["dataset_info.json", "state.json"],
    ],
)
def test_prepare_data_overwrite_replaces_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata: list[str],
):
    output = tmp_path / "out"
    _write_arrow_output(output, metadata)
    calls = {"load": 0, "save": 0}

    class _Dataset:
        def save_to_disk(self, path: str) -> None:
            calls["save"] += 1
            assert Path(path) == output
            (output / "data-00000-of-00001.arrow").write_text("fresh")
            (output / "dataset_info.json").write_text("{}")
            (output / "state.json").write_text("{}")

    def fake_load(*_args, **_kwargs):
        calls["load"] += 1
        assert output.is_dir()
        assert not (output / "data-00000-of-00001.arrow").exists()
        return _Dataset(), None

    monkeypatch.setattr(_prepare_data_module, "load_and_preprocess_dataset", fake_load)

    prepare_data(
        model="target-model",
        data=["conversations.jsonl"],
        output=str(output),
        overwrite=True,
    )

    assert calls == {"load": 1, "save": 1}
    assert (output / "data-00000-of-00001.arrow").read_text() == "fresh"
    assert (output / "dataset_info.json").is_file()
    assert (output / "state.json").is_file()


class _FakeProcessor:
    """Minimal processor stub that passes the chat-template precondition."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, *args, **kwargs):
        return ""


class _NoChatTemplateProcessor:
    chat_template = None


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
        preprocessing_module, "build_speculator_training_dataset", lambda *a, **k: empty
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


def test_pretokenized_data_does_not_require_chat_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    raw = HFDataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[0, 1, 1]],
        }
    )
    monkeypatch.setattr(
        preprocessing_module,
        "load_processor",
        lambda *a, **k: _NoChatTemplateProcessor(),
    )
    monkeypatch.setattr(
        preprocessing_module, "load_raw_dataset", lambda _path: (raw, None)
    )
    processed = HFDataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[0, 1, 1]],
            "seq_len": [3],
        }
    )
    processed.set_format(type="torch")
    monkeypatch.setattr(
        preprocessing_module,
        "build_speculator_training_dataset",
        lambda *a, **k: processed,
    )
    monkeypatch.setattr(
        preprocessing_module, "save_token_frequency_distribution", lambda **k: None
    )
    monkeypatch.setattr(preprocessing_module, "_visualize_sample", lambda *a, **k: None)

    dataset, _ = load_and_preprocess_dataset(
        "custom-model",
        ["pretokenized.jsonl"],
        seq_length=8,
        build_dataset_num_proc=1,
        token_freq_path=tmp_path / "token_freq.pt",
    )

    assert len(dataset) == 1
    assert dataset[0]["input_ids"].tolist() == [1, 2, 3]


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


def test_conversation_data_still_requires_chat_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Without a render endpoint, conversation data still fails fast when the
    processor exposes no chat template."""
    raw = HFDataset.from_dict(
        {
            "conversations": [
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            ]
        }
    )
    monkeypatch.setattr(
        preprocessing_module,
        "load_processor",
        lambda *a, **k: _NoChatTemplateProcessor(),
    )
    monkeypatch.setattr(
        preprocessing_module, "load_raw_dataset", lambda _path: (raw, None)
    )

    with pytest.raises(ValueError, match="does not support chat templates"):
        load_and_preprocess_dataset(
            "custom-model",
            ["conversations.jsonl"],
            seq_length=8,
            build_dataset_num_proc=1,
            token_freq_path=tmp_path / "token_freq.pt",
        )


def test_render_endpoint_bypasses_chat_template_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Server-side rendering needs no local chat template: the processor's
    template is never used when a render endpoint is provided."""
    raw = HFDataset.from_dict(
        {
            "conversations": [
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            ]
        }
    )
    monkeypatch.setattr(
        preprocessing_module,
        "load_processor",
        lambda *a, **k: _NoChatTemplateProcessor(),
    )
    monkeypatch.setattr(
        preprocessing_module, "load_raw_dataset", lambda _path: (raw, None)
    )
    processed = HFDataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[0, 1, 1]],
            "seq_len": [3],
        }
    )
    processed.set_format(type="torch")
    monkeypatch.setattr(
        preprocessing_module,
        "build_speculator_training_dataset",
        lambda *a, **k: processed,
    )
    monkeypatch.setattr(
        preprocessing_module, "save_token_frequency_distribution", lambda **k: None
    )
    monkeypatch.setattr(preprocessing_module, "_visualize_sample", lambda *a, **k: None)

    dataset, _ = load_and_preprocess_dataset(
        "custom-model",
        ["conversations.jsonl"],
        seq_length=8,
        build_dataset_num_proc=1,
        token_freq_path=tmp_path / "token_freq.pt",
        render_endpoint="http://localhost:8000",
    )

    assert len(dataset) == 1
