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


def _args(**overrides) -> Namespace:
    values = {
        "dataset": "org/dataset",
        "dataset_dir": None,
        "data_column_mapper": evaluate.DEFAULT_DATA_COLUMN_MAPPER,
        "subsets": "qa",
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_longbench_v2(tmp_path):
    path = tmp_path / "longbench_v2.jsonl"
    path.write_text('{"prompt": "text"}\n')

    assert evaluate._resolve_runs(
        _args(dataset="longbench-v2", dataset_dir=tmp_path)
    ) == [
        evaluate.RunSpec(
            "longbench-v2",
            str(path),
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
