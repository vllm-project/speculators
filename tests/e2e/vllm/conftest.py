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
        [{"role": "user", "content": "The president of United States is"}],
        [{"role": "user", "content": "The capital of France is"}],
        [{"role": "user", "content": "My name is"}],
    ]
