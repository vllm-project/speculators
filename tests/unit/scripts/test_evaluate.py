"""Tests for local benchmark resolution in scripts/evaluate/evaluate.py."""

import importlib
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "evaluate"
sys.path.insert(0, str(_SCRIPT_DIR))
try:
    evaluate = importlib.import_module("evaluate")
finally:
    sys.path.pop(0)


def _task(data_dir: Path, setup: str, name: str) -> Path:
    path = data_dir / setup / name / "test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"question": "prompt"}\n')
    return path


def test_resolve_ruler2_setup(tmp_path):
    paths = [
        _task(tmp_path, "model-32k", "mk_niah_basic"),
        _task(tmp_path, "model-32k", "qa_hard"),
    ]

    assert evaluate._resolve_ruler2("ruler2/model-32k", tmp_path) == [
        ("ruler2/model-32k/mk_niah_basic", paths[0]),
        ("ruler2/model-32k/qa_hard", paths[1]),
    ]


def test_resolve_ruler2_task(tmp_path):
    path = _task(tmp_path, "model-32k", "mv_niah_hard")

    assert evaluate._resolve_ruler2("ruler2/model-32k/mv_niah_hard", tmp_path) == [
        ("ruler2/model-32k/mv_niah_hard", path)
    ]


def test_resolve_ruler2_missing(tmp_path):
    with pytest.raises(SystemExit):
        evaluate._resolve_ruler2("ruler2/model-32k", tmp_path)
