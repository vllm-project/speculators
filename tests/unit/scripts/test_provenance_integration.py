"""Integration test: run launch_vllm.py --dry-run and verify provenance artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
LAUNCH_VLLM = str(SCRIPTS_DIR / "launch_vllm.py")
MODEL = "gpt2"
DRAFTER = "some-org/drafter-model"


@pytest.fixture
def provenance_dir(tmp_path: Path) -> Path:
    return tmp_path / "provenance"


class TestLaunchVllmProvenance:
    """Train subcommand integration tests."""

    def _run(
        self, provenance_dir: Path, *extra_args: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            [
                sys.executable,
                LAUNCH_VLLM,
                "train",
                MODEL,
                "--dry-run",
                "--provenance-dir",
                str(provenance_dir),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_creates_all_provenance_files(self, provenance_dir: Path):
        result = self._run(provenance_dir)
        assert result.returncode == 0, result.stderr
        assert (provenance_dir / "vllm_command.txt").exists()
        assert (provenance_dir / "vllm.patch").exists()
        assert (provenance_dir / "checkpoint_sha256.txt").exists()

    def test_vllm_command_contains_metadata(self, provenance_dir: Path):
        self._run(provenance_dir)
        content = (provenance_dir / "vllm_command.txt").read_text()
        assert "# Timestamp:" in content
        assert "# Python:" in content
        assert "# Git SHA:" in content
        assert "# vllm:" in content
        assert "vllm.entrypoints" in content

    def test_checkpoint_sha256_remote_model(self, provenance_dir: Path):
        self._run(provenance_dir)
        content = (provenance_dir / "checkpoint_sha256.txt").read_text()
        assert "not a local path" in content

    def test_default_provenance_dir_created(self, tmp_path: Path):
        result = subprocess.run(  # noqa: S603
            [sys.executable, LAUNCH_VLLM, "train", MODEL, "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        prov_dirs = [
            d
            for d in tmp_path.iterdir()
            if d.is_dir() and d.name.startswith("vllm_train_gpt2_")
        ]
        assert len(prov_dirs) == 1
        prov = prov_dirs[0]
        assert (prov / "vllm_command.txt").exists()
        assert (prov / "vllm.patch").exists()
        assert (prov / "checkpoint_sha256.txt").exists()

    def test_implicit_train_subcommand(self, provenance_dir: Path):
        """MODEL without 'train' prefix defaults to train mode."""
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                LAUNCH_VLLM,
                MODEL,
                "--dry-run",
                "--provenance-dir",
                str(provenance_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        content = (provenance_dir / "vllm_command.txt").read_text()
        assert "extract_hidden_states" in content

    def test_flags_only_falls_back_to_train(self):
        """Unknown flags without explicit subcommand fall back to train."""
        result = subprocess.run(  # noqa: S603
            [sys.executable, LAUNCH_VLLM, "--some-vllm-flag"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode != 0
        assert "model" in result.stderr.lower()

    def test_no_args_shows_usage_error(self):
        """Bare launch_vllm.py with no arguments reports missing model."""
        result = subprocess.run(  # noqa: S603
            [sys.executable, LAUNCH_VLLM],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode != 0
        assert "model" in result.stderr.lower()


class TestLaunchVllmEvalMode:
    """Eval subcommand integration tests."""

    def _run(
        self, provenance_dir: Path, *extra_args: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            [
                sys.executable,
                LAUNCH_VLLM,
                "eval",
                MODEL,
                "--spec-model",
                DRAFTER,
                "--spec-tokens",
                "3",
                "--dry-run",
                "--provenance-dir",
                str(provenance_dir),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_eval_mode_creates_provenance(self, provenance_dir: Path):
        result = self._run(provenance_dir)
        assert result.returncode == 0, result.stderr
        assert (provenance_dir / "vllm_command.txt").exists()
        assert (provenance_dir / "vllm.patch").exists()
        assert (provenance_dir / "checkpoint_sha256.txt").exists()
        assert (provenance_dir / "drafter_checkpoint_sha256.txt").exists()

    def test_eval_command_contains_spec_flags(self, provenance_dir: Path):
        self._run(provenance_dir)
        content = (provenance_dir / "vllm_command.txt").read_text()
        assert "--spec-model" in content
        assert DRAFTER in content
        assert "--spec-tokens" in content
        assert "3" in content

    def test_eval_command_no_train_flags(self, provenance_dir: Path):
        self._run(provenance_dir)
        content = (provenance_dir / "vllm_command.txt").read_text()
        assert "--speculative_config" not in content
        assert "--kv_transfer_config" not in content
        assert "extract_hidden_states" not in content

    def test_eval_mode_requires_spec_model(self, tmp_path: Path):
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                LAUNCH_VLLM,
                "eval",
                MODEL,
                "--dry-run",
                "--provenance-dir",
                str(tmp_path / "prov"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode != 0
        assert "--spec-model" in result.stderr

    def test_eval_mode_with_spec_method(self, provenance_dir: Path):
        result = self._run(provenance_dir, "--spec-method", "dflash")
        assert result.returncode == 0, result.stderr
        content = (provenance_dir / "vllm_command.txt").read_text()
        assert "--spec-method" in content
        assert "dflash" in content

    def test_eval_mode_passes_extra_vllm_args(self, provenance_dir: Path):
        result = self._run(provenance_dir, "--", "--port", "8080", "--tp", "2")
        assert result.returncode == 0, result.stderr
        content = (provenance_dir / "vllm_command.txt").read_text()
        assert "--port" in content
        assert "8080" in content
        assert "--tp" in content
