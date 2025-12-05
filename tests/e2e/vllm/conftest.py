import pytest


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(cache_dir))
    return cache_dir


@pytest.fixture
def prompts():
    return [
        "The capital of France is",
        "The president of the US is",
        "My name is",
    ]
