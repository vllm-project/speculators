from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))


from evaluate import (  # type: ignore[import-not-found]
    _git_sha,
    _speculators_repo_root,
    save_eval_provenance,
)


class TestSpeculatorsRepoRoot:
    def test_returns_string_when_found(self):
        root = _speculators_repo_root()
        if root is not None:
            assert isinstance(root, str)
            assert (Path(root) / ".git").is_dir()

    def test_returns_none_when_import_fails(self):
        with patch.dict(sys.modules, {"speculators": None}):
            assert _speculators_repo_root() is None


class TestGitSha:
    def test_returns_hex_string(self):
        sha = _git_sha()
        is_valid_hex = len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
        assert sha == "unknown" or is_valid_hex

    def test_uses_speculators_repo_root(self):
        with (
            patch(
                "evaluate._speculators_repo_root", return_value="/fake/repo"
            ) as mock_root,
            patch("evaluate.subprocess.run") as mock_run,
        ):
            mock_run.return_value.stdout = "abc123\n"
            mock_run.return_value.returncode = 0
            _git_sha()
            mock_root.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs["cwd"] == "/fake/repo"

    def test_fallback_on_oserror(self):
        with patch("evaluate.subprocess.run", side_effect=OSError("no git")):
            assert _git_sha() == "unknown"


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
        with patch("evaluate._atomic_write", side_effect=RuntimeError("disk full")):
            save_eval_provenance(tmp_path)

    def test_overwrites_existing(self, tmp_path: Path):
        (tmp_path / "eval_command.txt").write_text("old content")
        save_eval_provenance(tmp_path)
        content = (tmp_path / "eval_command.txt").read_text()
        assert "old content" not in content
        assert "# Timestamp:" in content
