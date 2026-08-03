"""Tests for local benchmark resolution in scripts/evaluate/evaluate.py."""

import importlib
import sys
from argparse import Namespace
from pathlib import Path

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


def _args(**overrides) -> Namespace:
    values = {
        "dataset": "org/dataset",
        "dataset_dir": None,
        "data_column_mapper": evaluate.DEFAULT_DATA_COLUMN_MAPPER,
        "subsets": "qa",
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_ruler2_setup_and_task(tmp_path):
    paths = [
        _task(tmp_path, "model-32k", "mk_niah_basic"),
        _task(tmp_path, "model-32k", "qa_hard"),
    ]

    assert evaluate._resolve_ruler2("ruler2/model-32k", tmp_path) == [
        ("ruler2/model-32k/mk_niah_basic", paths[0]),
        ("ruler2/model-32k/qa_hard", paths[1]),
    ]
    assert evaluate._resolve_ruler2("ruler2/model-32k/qa_hard", tmp_path) == [
        ("ruler2/model-32k/qa_hard", paths[1])
    ]


def test_resolve_longbench_all_and_task(tmp_path):
    paths = [
        tmp_path / "longbench_hotpotqa.jsonl",
        tmp_path / "longbench_qasper.jsonl",
    ]
    for path in paths:
        path.write_text('{"prompt": "text"}\n')

    assert evaluate._resolve_longbench("longbench", tmp_path) == [
        ("longbench/hotpotqa", paths[0]),
        ("longbench/qasper", paths[1]),
    ]
    assert evaluate._resolve_runs(
        _args(dataset="longbench/qasper", dataset_dir=tmp_path)
    ) == [
        evaluate.RunSpec(
            "longbench/qasper",
            str(paths[1]),
            evaluate.DEFAULT_DATA_COLUMN_MAPPER,
        )
    ]


def test_resolve_runs_distinguishes_hf_and_local(tmp_path):
    assert evaluate._resolve_runs(_args()) == [
        evaluate.RunSpec(
            "qa",
            "org/dataset",
            evaluate.DEFAULT_DATA_COLUMN_MAPPER,
            "qa",
        ),
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "text"}\n')

    assert evaluate._resolve_runs(_args(dataset=str(path))) == [
        evaluate.RunSpec(
            "prompts",
            str(path),
            evaluate.DEFAULT_DATA_COLUMN_MAPPER,
        )
    ]


def test_resolve_runs_preserves_speedbench(tmp_path):
    path = tmp_path / "qualitative_coding.jsonl"
    path.write_text('{"turns": "prompt"}\n')

    assert evaluate._resolve_runs(
        _args(dataset="speedbench/qualitative/coding", dataset_dir=tmp_path)
    ) == [
        evaluate.RunSpec(
            "speedbench/qualitative/coding",
            str(path),
            evaluate._SPEEDBENCH_COLUMN_MAPPER,
        )
    ]
