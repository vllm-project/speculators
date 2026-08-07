from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


from launch_vllm import (  # type: ignore[import-not-found]
    _find_vllm_repo,
    _is_vllm_repo,
    _save_checkpoint_sha256,
    _save_vllm_command,
    _save_vllm_patch,
    _save_vllm_provenance,
)


class TestIsVllmRepo:
    def test_returns_true_for_valid_repo(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "vllm").mkdir()
        (tmp_path / "vllm" / "__init__.py").touch()
        assert _is_vllm_repo(str(tmp_path))

    def test_returns_false_without_git(self, tmp_path: Path):
        (tmp_path / "vllm").mkdir()
        (tmp_path / "vllm" / "__init__.py").touch()
        assert not _is_vllm_repo(str(tmp_path))

    def test_returns_false_without_vllm_package(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        assert not _is_vllm_repo(str(tmp_path))


class TestFindVllmRepo:
    def test_returns_string_or_none(self):
        result = _find_vllm_repo()
        assert result is None or isinstance(result, str)

    def test_returns_none_when_vllm_not_importable(self):
        with patch("launch_vllm.importlib.util.find_spec", return_value=None):
            assert _find_vllm_repo() is None


class TestSaveVllmCommand:
    def test_creates_file(self, tmp_path: Path):
        _save_vllm_command(
            str(tmp_path),
            ["python", "-m", "vllm", "serve", "model"],
            "abc123",
            "",
            "0.1.0",
        )
        assert (tmp_path / "vllm_command.txt").exists()

    def test_contains_metadata(self, tmp_path: Path):
        _save_vllm_command(
            str(tmp_path),
            ["python", "-m", "vllm", "serve", "model"],
            "abc123",
            "",
            "0.1.0",
        )
        content = (tmp_path / "vllm_command.txt").read_text()
        assert "# Timestamp:" in content
        assert "# Python:" in content
        assert "# Git SHA: abc123" in content
        assert "# vllm: 0.1.0" in content
        assert "python -m vllm serve model" in content

    def test_dirty_marker_when_diff_present(self, tmp_path: Path):
        _save_vllm_command(str(tmp_path), ["cmd"], "abc123", "some diff", "0.1.0")
        content = (tmp_path / "vllm_command.txt").read_text()
        assert "abc123 (dirty)" in content

    def test_no_dirty_marker_when_clean(self, tmp_path: Path):
        _save_vllm_command(str(tmp_path), ["cmd"], "abc123", "", "0.1.0")
        content = (tmp_path / "vllm_command.txt").read_text()
        assert "(dirty)" not in content


class TestSaveVllmPatch:
    def test_records_repo_and_diff(self, tmp_path: Path):
        _save_vllm_patch(
            str(tmp_path), "/path/to/vllm", "abc123", "diff content", "0.1.0"
        )
        content = (tmp_path / "vllm.patch").read_text()
        assert "# repo: /path/to/vllm (abc123)" in content
        assert "diff content" in content

    def test_header_only_for_clean_checkout(self, tmp_path: Path):
        _save_vllm_patch(str(tmp_path), "/path/to/vllm", "abc123", "", "0.1.0")
        content = (tmp_path / "vllm.patch").read_text()
        assert "# repo: /path/to/vllm (abc123)" in content

    def test_wheel_install_fallback(self, tmp_path: Path):
        _save_vllm_patch(str(tmp_path), None, "unknown", "", "0.5.0")
        content = (tmp_path / "vllm.patch").read_text()
        assert "wheel install" in content
        assert "0.5.0" in content


class TestSaveCheckpointSha256:
    def test_remote_model(self, tmp_path: Path):
        _save_checkpoint_sha256(str(tmp_path), "org/model-name")
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "not a local path" in content

    def test_local_model_with_safetensors(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake weights")
        _save_checkpoint_sha256(str(tmp_path), str(model_dir))
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "model.safetensors" in content
        assert len(content.split("  ")[0]) == 64  # SHA256 hex length

    def test_no_safetensors(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        _save_checkpoint_sha256(str(tmp_path), str(model_dir))
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "no .safetensors files" in content

    def test_skip_hash_records_size_and_mtime(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake weights")
        _save_checkpoint_sha256(str(tmp_path), str(model_dir), skip_hash=True)
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "# hashing skipped (--no-hash-checkpoints)" in content
        assert "size=" in content
        assert "mtime=" in content
        assert "model.safetensors" in content

    def test_skip_hash_does_not_contain_sha256(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake weights")
        _save_checkpoint_sha256(str(tmp_path), str(model_dir), skip_hash=True)
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        data_lines = [line for line in content.splitlines() if not line.startswith("#")]
        for line in data_lines:
            assert line.startswith("size=")

    def test_multiple_safetensors_sorted(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for name in ("c.safetensors", "a.safetensors", "b.safetensors"):
            (model_dir / name).write_bytes(b"data")
        _save_checkpoint_sha256(str(tmp_path), str(model_dir))
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        names = [line.split("  ")[1] for line in content.strip().splitlines()]
        assert names == ["a.safetensors", "b.safetensors", "c.safetensors"]


class TestSaveVllmProvenance:
    def test_creates_all_artifacts(self, tmp_path: Path):
        prov_dir = tmp_path / "provenance"
        _save_vllm_provenance(["python", "serve"], str(prov_dir), "org/model")
        assert (prov_dir / "vllm_command.txt").exists()
        assert (prov_dir / "vllm.patch").exists()
        assert (prov_dir / "checkpoint_sha256.txt").exists()

    def test_creates_provenance_dir(self, tmp_path: Path):
        prov_dir = tmp_path / "nested" / "provenance"
        _save_vllm_provenance(["cmd"], str(prov_dir), "org/model")
        assert prov_dir.is_dir()

    def test_bad_dir_warns_and_returns(self, tmp_path: Path, capsys):
        _save_vllm_provenance(["cmd"], "/dev/null/impossible", "org/model")
        captured = capsys.readouterr()
        assert "Warning:" in captured.err

    def test_skip_hash_propagated(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.safetensors").write_bytes(b"data")
        prov_dir = tmp_path / "prov"
        _save_vllm_provenance(["cmd"], str(prov_dir), str(model_dir), skip_hash=True)
        content = (prov_dir / "checkpoint_sha256.txt").read_text()
        assert "hashing skipped" in content

    def test_individual_writer_failure_does_not_block_others(self, tmp_path: Path):
        prov_dir = tmp_path / "prov"
        with patch("launch_vllm._save_vllm_command", side_effect=RuntimeError("boom")):
            _save_vllm_provenance(["cmd"], str(prov_dir), "org/model")
        assert not (prov_dir / "vllm_command.txt").exists()
        assert (prov_dir / "vllm.patch").exists()
        assert (prov_dir / "checkpoint_sha256.txt").exists()
