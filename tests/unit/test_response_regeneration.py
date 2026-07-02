"""Unit tests for scripts/response_regeneration/script.py resume handling."""

import importlib.util
import json
from pathlib import Path

import pytest

# The script imports optional, scripts-only deps at module load.
pytest.importorskip("aiohttp")
pytest.importorskip("datasets")

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "response_regeneration"
    / "script.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("regen_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_seen_matches_written_ids(tmp_path):
    """load_seen must key off the same ``id`` the script writes, and must not
    mark errored rows as seen (so they are retried on resume)."""
    mod = _load_module()
    out = tmp_path / "out.jsonl"
    records = [
        # uuid-less dataset: written id is f"sample_{idx}"
        {"id": "sample_5", "metadata": {"idx": 5, "finish_reason": "stop"}},
        # uuid dataset: written id is the uuid
        {"id": "abc-123", "metadata": {"idx": 9, "finish_reason": "stop"}},
        # errored row: must NOT be seen -> retried
        {"id": "sample_7", "metadata": {"idx": 7, "error": "boom"}},
    ]
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    seen = mod.load_seen(str(out))

    assert seen == {"sample_5", "abc-123"}
    assert "sample_7" not in seen


def test_load_seen_missing_file(tmp_path):
    mod = _load_module()
    assert mod.load_seen(str(tmp_path / "does_not_exist.jsonl")) == set()
