from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))


from evaluate import (  # type: ignore[import-not-found]
    main,
    save_eval_provenance,
)


class TestDatasetLoadKwargs:
    @pytest.fixture
    def cli_args(self, tmp_path, monkeypatch):
        argv = [
            "evaluate.py",
            "throughput",
            "--target",
            "http://localhost:8000/v1",
            "--output-dir",
            str(tmp_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        return argv

    @pytest.mark.parametrize(
        "load_kwargs", [None, {}, {"name": "clean", "split": "test[:3]"}]
    )
    def test_cli(self, cli_args, load_kwargs):
        cli_args += [
            "--data-preprocessor",
            "kind=encode_media,audio_kwargs.audio_format=wav",
        ]
        if load_kwargs is not None:
            cli_args.extend(["--dataset-load-kwargs", json.dumps(load_kwargs)])
        with patch("evaluate.run_benchmark") as run:
            main()
        assert run.call_args.args[0].dataset_load_kwargs == load_kwargs
        assert run.call_args.args[0].data_preprocessor == [
            "kind=encode_media,audio_kwargs.audio_format=wav"
        ]

    @pytest.mark.parametrize("value", ["{", "[]", "null", '"clean"'])
    def test_rejects_invalid_object(self, cli_args, capsys, value):
        cli_args.extend(["--dataset-load-kwargs", value])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "--dataset-load-kwargs" in capsys.readouterr().err

    @pytest.mark.parametrize("dataset", ["openai/mrcr", "speedbench/qualitative"])
    def test_rejects_specialized_datasets(self, cli_args, capsys, dataset):
        cli_args.extend(["--dataset", dataset, "--dataset-load-kwargs", "{}"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "not supported for MRCR/SPEED-Bench" in capsys.readouterr().err


class TestSaveEvalProvenance:
    def test_creates_file(self, tmp_path: Path):
        save_eval_provenance(tmp_path)
        assert (tmp_path / "eval_command.txt").exists()

    def test_contains_timestamp(self, tmp_path: Path):
        save_eval_provenance(tmp_path)
        content = (tmp_path / "eval_command.txt").read_text()
        assert "# Timestamp:" in content

    def test_contains_git_sha(self, tmp_path: Path):
        save_eval_provenance(tmp_path)
        content = (tmp_path / "eval_command.txt").read_text()
        assert "# Git SHA:" in content

    def test_contains_package_versions(self, tmp_path: Path):
        save_eval_provenance(tmp_path)
        content = (tmp_path / "eval_command.txt").read_text()
        pkgs = ("speculators", "vllm", "transformers", "torch", "compressed-tensors")
        for pkg in pkgs:
            assert f"# {pkg}:" in content

    def test_contains_sys_argv(self, tmp_path: Path):
        argv = ["evaluate.py", "--target", "http://localhost:8000/v1"]
        with patch.object(sys, "argv", argv):
            save_eval_provenance(tmp_path)
        content = (tmp_path / "eval_command.txt").read_text()
        assert "evaluate.py --target" in content

    def test_no_leftover_tmp_files(self, tmp_path: Path):
        save_eval_provenance(tmp_path)
        tmp_files = [
            f for f in tmp_path.iterdir() if f.name.startswith(".eval_command")
        ]
        assert tmp_files == []

    def test_best_effort_never_raises(self, tmp_path: Path):
        with patch("evaluate.atomic_write", side_effect=OSError("disk full")):
            save_eval_provenance(tmp_path)

    def test_overwrites_existing(self, tmp_path: Path):
        (tmp_path / "eval_command.txt").write_text("old content")
        save_eval_provenance(tmp_path)
        content = (tmp_path / "eval_command.txt").read_text()
        assert "old content" not in content
        assert "# Timestamp:" in content
