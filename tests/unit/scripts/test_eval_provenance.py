from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))


from evaluate import (  # type: ignore[import-not-found]
    save_eval_provenance,
)


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
