"""Tests for scripts/ci/push_metrics.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "push_metrics.py"


def _run_git(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _git_config(cwd: Path) -> None:
    _run_git(["config", "user.name", "test"], cwd=str(cwd))
    _run_git(["config", "user.email", "test@test"], cwd=str(cwd))


def _load_module():
    spec = importlib.util.spec_from_file_location("push_metrics", str(SCRIPT))
    assert spec is not None, f"no import spec for {SCRIPT}"
    assert spec.loader is not None, f"no loader for {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_repo(tmp_path: Path):
    """Create a bare git repo with a metrics-data branch."""
    bare = tmp_path / "remote.git"
    _run_git(["init", "--bare", str(bare)])

    work = tmp_path / "work"
    work.mkdir()
    _run_git(["init", str(work)])
    _run_git(["remote", "add", "origin", str(bare)], cwd=str(work))
    _git_config(work)
    _run_git(["checkout", "--orphan", "metrics-data"], cwd=str(work))
    (work / "README.md").write_text("init")
    _run_git(["add", "README.md"], cwd=str(work))
    _run_git(["commit", "-m", "init"], cwd=str(work))
    _run_git(["push", "origin", "metrics-data"], cwd=str(work))
    return bare, work


def test_push_single_record(fake_repo, tmp_path: Path):
    bare, work = fake_repo

    record = {"timestamp": "2026-01-01T00:00:00Z", "value": 42}
    mod = _load_module()

    original = mod._repo_root
    mod._repo_root = lambda: work
    try:
        mod._push_records([record], "test.json", "metrics-data")
    finally:
        mod._repo_root = original

    verify = tmp_path / "verify"
    _run_git(["clone", "--branch", "metrics-data", str(bare), str(verify)])

    records = json.loads((verify / "test.json").read_text())
    assert len(records) == 1
    assert records[0]["value"] == 42


def test_push_appends_to_existing(fake_repo, tmp_path: Path):
    bare, work = fake_repo
    mod = _load_module()

    original = mod._repo_root
    mod._repo_root = lambda: work
    try:
        mod._push_records([{"idx": 0}], "data.json", "metrics-data")
        mod._push_records([{"idx": 1}, {"idx": 2}], "data.json", "metrics-data")
    finally:
        mod._repo_root = original

    verify = tmp_path / "verify"
    _run_git(["clone", "--branch", "metrics-data", str(bare), str(verify)])

    records = json.loads((verify / "data.json").read_text())
    assert len(records) == 3
    assert [r["idx"] for r in records] == [0, 1, 2]


def test_push_dir(fake_repo, tmp_path: Path):
    bare, work = fake_repo

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for i in range(3):
        (metrics_dir / f"r{i}.json").write_text(json.dumps({"idx": i}))

    mod = _load_module()
    records: list[dict] = []
    for p in sorted(metrics_dir.glob("*.json")):
        records.append(json.loads(p.read_text()))

    original = mod._repo_root
    mod._repo_root = lambda: work
    try:
        mod._push_records(records, "batch.json", "metrics-data")
    finally:
        mod._repo_root = original

    verify = tmp_path / "verify"
    _run_git(["clone", "--branch", "metrics-data", str(bare), str(verify)])

    result = json.loads((verify / "batch.json").read_text())
    assert len(result) == 3
    assert [r["idx"] for r in result] == [0, 1, 2]
