import os
from pathlib import Path

import pytest


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        cache_dir = Path(os.environ["HF_HOME"]) / "hub"
        assert cache_dir.is_dir(), "frozen attempt-local Hub cache is missing"
        return cache_dir
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(cache_dir))
    return cache_dir


@pytest.fixture
def prompts():
    return [
        [{"role": "user", "content": "Write a binary search function in python"}],
        [{"role": "user", "content": "Explain how speculative decoding works"}],
        [{"role": "user", "content": "Code a transformer block function"}],
    ]
