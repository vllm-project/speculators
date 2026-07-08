from collections.abc import Generator

import pytest
from loguru import logger


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
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


@pytest.fixture
def log_perf(request: pytest.FixtureRequest) -> Generator[dict[str, float], None, None]:
    """Collect per-stage wall-clock timings and log them after the test.

    Usage in a test:

        def test_something(log_perf):
            with record_perf("training", log_perf):
                run_training(...)
            with record_perf("vllm_inference", log_perf):
                run_vllm_engine(...)
    """
    results: dict[str, float] = {}
    yield results

    if not results:
        return

    lines = "\n".join(
        f"  {label}: {elapsed:.1f}s" for label, elapsed in results.items()
    )
    logger.info("Performance timings for {}:\n{}", request.node.name, lines)
