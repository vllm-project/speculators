from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from speculators.provenance import (
    atomic_write,
    find_package_repo,
    find_repo_root,
    git_diff,
    git_sha,
    package_versions,
    pkg_version,
    run_git,
)


class TestAtomicWrite:
    def test_creates_file(self, tmp_path: Path):
        dest = tmp_path / "out.txt"
        atomic_write(dest, "hello")
        assert dest.read_text() == "hello"

    def test_overwrites_existing(self, tmp_path: Path):
        dest = tmp_path / "out.txt"
        dest.write_text("old")
        atomic_write(dest, "new")
        assert dest.read_text() == "new"

    def test_no_leftover_tmp_files(self, tmp_path: Path):
        dest = tmp_path / "out.txt"
        atomic_write(dest, "data")
        tmp_files = [f for f in tmp_path.iterdir() if f.name.startswith(".")]
        assert tmp_files == []


class TestPkgVersion:
    def test_known_package(self):
        assert pkg_version("speculators") != "not installed"

    def test_unknown_package(self):
        assert pkg_version("nonexistent_pkg_xyz") == "not installed"


class TestFindRepoRoot:
    def test_finds_from_file(self):
        root = find_repo_root(Path(__file__))
        assert root is not None
        assert (root / ".git").exists()

    def test_finds_from_directory(self):
        root = find_repo_root(Path(__file__).parent)
        assert root is not None
        assert (root / ".git").exists()

    def test_returns_none_for_root(self):
        assert find_repo_root(Path("/")) is None


class TestFindPackageRepo:
    def test_finds_speculators(self):
        root = find_package_repo("speculators")
        if root is not None:
            assert (root / ".git").exists()

    def test_returns_none_for_missing(self):
        assert find_package_repo("nonexistent_pkg_xyz") is None


class TestGitSha:
    def test_returns_sha_or_unknown(self):
        sha = git_sha(find_repo_root(Path(__file__)))
        is_hex = len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
        assert sha == "unknown" or is_hex

    def test_none_returns_unknown(self):
        assert git_sha(None) == "unknown"

    def test_fallback_on_oserror(self):
        with patch("speculators.provenance.subprocess.run", side_effect=OSError):
            assert git_sha(Path("/tmp")) == "unknown"

    def test_fallback_on_timeout(self):
        with patch(
            "speculators.provenance.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 5),
        ):
            assert git_sha(Path("/tmp")) == "unknown"


class TestGitDiff:
    def test_none_returns_empty(self):
        assert git_diff(None) == ""

    def test_returns_string(self):
        root = find_repo_root(Path(__file__))
        result = git_diff(root)
        assert isinstance(result, str)

    def test_fallback_on_oserror(self):
        with patch("speculators.provenance.subprocess.run", side_effect=OSError):
            assert git_diff(Path("/tmp")) == ""


class TestRunGit:
    def test_rev_parse(self):
        root = find_repo_root(Path(__file__))
        if root:
            sha = run_git(["git", "rev-parse", "HEAD"], root)
            assert len(sha) == 40

    def test_failure_returns_empty(self):
        assert run_git(["git", "no-such-command"], ".") == ""

    def test_oserror_returns_empty(self):
        with patch("speculators.provenance.subprocess.run", side_effect=OSError):
            assert run_git(["git", "status"], ".") == ""


class TestPackageVersions:
    def test_returns_header_lines(self):
        lines = package_versions(("speculators",))
        assert len(lines) == 1
        assert lines[0].startswith("# speculators:")

    def test_default_packages(self):
        lines = package_versions()
        assert len(lines) == 5
