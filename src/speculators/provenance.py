"""Shared provenance helpers for reproducibility artifacts.

Used by training, evaluation, and vLLM-launch scripts to record
command lines, git state, and package versions.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

TRACKED_PACKAGES = (
    "speculators",
    "vllm",
    "transformers",
    "torch",
    "compressed-tensors",
)


def atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via tempfile + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}_", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def pkg_version(name: str) -> str:
    """Return the installed version of *name*, or ``'not installed'``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def find_repo_root(start: Path) -> Path | None:
    """Walk up from *start* to the nearest directory containing ``.git``."""
    try:
        d = start.resolve()
        if d.is_file():
            d = d.parent
        while d != d.parent:
            if (d / ".git").exists():
                return d
            d = d.parent
    except OSError:
        pass
    return None


def find_package_repo(package_name: str) -> Path | None:
    """Find the git repo root for an installed editable package."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return find_repo_root(Path(spec.origin))
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def git_sha(repo_root: Path | None) -> str:
    """Return the HEAD SHA of *repo_root*, or ``'unknown'`` on failure."""
    if repo_root is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def git_diff(repo_root: Path | None, *, timeout: int = 30) -> str:
    """Return ``git diff HEAD`` for *repo_root*, or empty string on failure."""
    if repo_root is None:
        return ""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def run_git(args: list[str], cwd: str | Path, *, timeout: int = 5) -> str:
    """Run a git command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> list[str]:
    """Return ``['# pkg: version', ...]`` header lines for *packages*."""
    return [f"# {p}: {pkg_version(p)}" for p in packages]
