"""Integration tests for launch_vllm.py provenance artifacts.

Includes a venv-isolation test that verifies the script imports successfully
even when `speculators` is not on sys.path — reproducing the exact failure
mode that caused PR #958 to be reverted in PR #1008.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
LAUNCH_VLLM = str(SCRIPTS_DIR / "launch_vllm.py")
MODEL = "gpt2"


@pytest.fixture
def provenance_dir(tmp_path: Path) -> Path:
    return tmp_path / "provenance"


class TestVenvIsolation:
    """launch_vllm.py must import without speculators installed (vLLM venv)."""

    def test_imports_without_speculators_on_path(self):
        """No ImportError when speculators is absent — the root cause of #1008."""
        # Strip path entries that would let `speculators` be found, then import.
        snippet = (
            "import sys; "
            "sys.path = [p for p in sys.path "
            "if 'speculators' not in p and 'site-packages' not in p]; "
            f"sys.path.insert(0, {str(SCRIPTS_DIR)!r}); "
            "import launch_vllm"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            f"launch_vllm.py raised ImportError when speculators was absent.\n"
            f"stderr: {result.stderr}"
        )


class TestLaunchVllmProvenance:
    def _run(
        self, provenance_dir: Path, *extra_args: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603
            [
                sys.executable,
                LAUNCH_VLLM,
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
            [sys.executable, LAUNCH_VLLM, MODEL, "--dry-run"],
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
            if d.is_dir() and d.name.startswith("vllm_gpt2_")
        ]
        assert len(prov_dirs) == 1
        prov = prov_dirs[0]
        assert (prov / "vllm_command.txt").exists()
        assert (prov / "vllm.patch").exists()
        assert (prov / "checkpoint_sha256.txt").exists()
