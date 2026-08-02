"""Tests for scripts/evaluate/prepare_longbench.py."""

import importlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "evaluate"
sys.path.insert(0, str(_SCRIPT_DIR))
try:
    prepare_longbench = importlib.import_module("prepare_longbench")
finally:
    sys.path.pop(0)


def test_prepare_renders_official_prompt_template(tmp_path):
    archive_path = tmp_path / "data.zip"
    prompts_path = tmp_path / "dataset2prompt.json"
    prompts_path.write_text(
        json.dumps({"qasper": "Article: {context}\nQuestion: {input}"})
    )
    row = {"context": "A long article", "input": "What happened?"}
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("data/qasper.jsonl", json.dumps(row))

    assert (
        prepare_longbench.prepare(
            archive_path,
            prompts_path,
            tmp_path,
        )
        == 1
    )
    output = tmp_path / "longbench_qasper.jsonl"
    assert json.loads(output.read_text()) == {
        "prompt": "Article: A long article\nQuestion: What happened?"
    }
