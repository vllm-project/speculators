"""Tests for scripts/evaluate/prepare_longbench_v2.py."""

import importlib
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "evaluate"
sys.path.insert(0, str(_SCRIPT_DIR))
try:
    prepare_longbench_v2 = importlib.import_module("prepare_longbench_v2")
finally:
    sys.path.pop(0)


def test_prepare_renders_official_prompt(tmp_path):
    source = tmp_path / "data.json"
    source.write_text(
        json.dumps(
            [
                {
                    "context": "A long document",
                    "question": "What happened?",
                    "choice_A": "One",
                    "choice_B": "Two",
                    "choice_C": "Three",
                    "choice_D": "Four",
                }
            ]
        )
    )
    output = tmp_path / "longbench_v2.jsonl"

    assert prepare_longbench_v2.prepare(source, output) == 1
    assert json.loads(output.read_text()) == {
        "prompt": (
            "Please read the following text and answer the question below.\n\n"
            "<text>\nA long document\n</text>\n\n"
            "What is the correct answer to this question: What happened?\n"
            "Choices:\n(A) One\n(B) Two\n(C) Three\n(D) Four\n\n"
            'Format your response as follows: "The correct answer is '
            '(insert answer here)".'
        )
    }
