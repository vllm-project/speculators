import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.utils import save_train_command

# ---------------------------------------------------------------------------
# save_train_command tests
# ---------------------------------------------------------------------------


class TestSaveTrainCommand:
    def test_creates_file(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        assert (tmp_path / "train_command.txt").exists()

    def test_creates_directory_if_missing(self, tmp_path: Path):
        save_path = tmp_path / "nested" / "dir"
        save_train_command(str(save_path))
        assert (save_path / "train_command.txt").exists()

    def test_contains_sys_argv(self, tmp_path: Path):
        with patch.object(sys, "argv", ["scripts/train.py", "--lr", "1e-4"]):
            save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "scripts/train.py --lr 1e-4" in content

    def test_header_has_timestamp(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "# Timestamp:" in content

    def test_header_has_git_sha(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "# Git SHA:" in content

    def test_header_has_world_size(self, tmp_path: Path):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "# World size: 8" in content

    def test_world_size_defaults_to_1(self, tmp_path: Path):
        env = os.environ.copy()
        env.pop("WORLD_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "# World size: 1" in content

    def test_header_has_package_versions(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        for pkg in ("speculators", "transformers", "torch"):
            assert f"# {pkg}:" in content

    def test_git_sha_fallback_on_error(self, tmp_path: Path):
        with patch(
            "speculators.train.utils._git_sha",
            return_value="unknown",
        ):
            save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "# Git SHA: unknown" in content

    def test_quotes_args_with_spaces(self, tmp_path: Path):
        with patch.object(sys, "argv", ["train.py", "--path", "/has spaces/dir"]):
            save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "'/has spaces/dir'" in content

    def test_no_leftover_tmp_files(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        tmp_files = [
            f for f in tmp_path.iterdir() if f.name.startswith(".train_command_")
        ]
        assert tmp_files == []

    def test_overwrites_existing(self, tmp_path: Path):
        (tmp_path / "train_command.txt").write_text("old content")
        save_train_command(str(tmp_path))
        content = (tmp_path / "train_command.txt").read_text()
        assert "old content" not in content
        assert "# Timestamp:" in content


# ---------------------------------------------------------------------------
# speculators.patch tests
# ---------------------------------------------------------------------------


class TestSpeculatorsPatch:
    def test_creates_patch_file(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        assert (tmp_path / "speculators.patch").exists()

    def test_patch_contains_repo_header(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        content = (tmp_path / "speculators.patch").read_text()
        assert content.startswith("# repo: ")

    def test_patch_contains_sha(self, tmp_path: Path):
        save_train_command(str(tmp_path))
        content = (tmp_path / "speculators.patch").read_text()
        first_line = content.split("\n")[0]
        assert "(" in first_line
        assert ")" in first_line

    def test_no_patch_when_no_repo(self, tmp_path: Path):
        with patch(
            "speculators.train.utils.find_repo_root",
            return_value=None,
        ):
            save_train_command(str(tmp_path))
        assert not (tmp_path / "speculators.patch").exists()

    def test_patch_failure_does_not_block(self, tmp_path: Path):
        original_run = subprocess.run

        def _fail_on_diff(*args, **kwargs):
            if "diff" in args[0]:
                raise OSError("git broke")
            return original_run(*args, **kwargs)

        with patch(
            "speculators.train.utils.subprocess.run",
            side_effect=_fail_on_diff,
        ):
            save_train_command(str(tmp_path))
        assert (tmp_path / "train_command.txt").exists()
        assert not (tmp_path / "speculators.patch").exists()


# ---------------------------------------------------------------------------
# _copy_train_command tests (checkpointer)
# ---------------------------------------------------------------------------


class TestCopyTrainCommand:
    def test_copies_into_epoch_dir(self, tmp_path: Path):
        src_content = "# test content\ntrain.py --lr 1e-4\n"
        (tmp_path / "train_command.txt").write_text(src_content)
        (tmp_path / "0").mkdir()

        cp = SingleGPUCheckpointer(str(tmp_path))
        cp._copy_train_command(0)

        copied = tmp_path / "0" / "train_command.txt"
        assert copied.exists()
        assert copied.read_text() == src_content

    def test_noop_when_source_missing(self, tmp_path: Path):
        (tmp_path / "0").mkdir()

        cp = SingleGPUCheckpointer(str(tmp_path))
        cp._copy_train_command(0)

        assert not (tmp_path / "0" / "train_command.txt").exists()

    def test_copies_into_string_epoch(self, tmp_path: Path):
        (tmp_path / "train_command.txt").write_text("content")
        (tmp_path / "interrupted").mkdir()

        cp = SingleGPUCheckpointer(str(tmp_path))
        cp._copy_train_command("interrupted")

        assert (tmp_path / "interrupted" / "train_command.txt").exists()

    def test_cleanup_keep_only_best_preserves_train_command(self, tmp_path: Path):
        (tmp_path / "train_command.txt").write_text("content")
        (tmp_path / "0").mkdir()
        (tmp_path / "1").mkdir()
        (tmp_path / "1" / "model.safetensors").touch()

        cp = SingleGPUCheckpointer(str(tmp_path))
        cp.update_best_symlink(1)
        cp.cleanup_keep_only_best(1)

        assert (tmp_path / "train_command.txt").exists()
        assert not (tmp_path / "0").exists()
