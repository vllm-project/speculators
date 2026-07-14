import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from PIL import Image

from scripts import pipeline_runners
from speculators.data_generation.preprocessing import load_raw_dataset


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


def purge_newfiles(fn: Callable[..., Path]):
    """Decorator that turns a Path-returning function into a context manager.

    On exit, deletes top-level files in the resolved directory whose mtime is
    newer than when the wrapped function returned.  Does not recurse into
    subdirectories.  This prevents generated artifacts (e.g. ``d2t.npy``,
    ``t2d.npy`` potentially cached by ``train.py``) from persisting in
    shared directories (such as the HF snapshot cache) between test runs.
    """

    @wraps(fn)
    @contextmanager
    def wrapper(*args, **kwargs):
        path = fn(*args, **kwargs)
        cutoff = time.time()
        try:
            yield path
        finally:
            if path.is_dir():
                for f in path.iterdir():
                    if f.is_file() and f.stat().st_mtime > cutoff:
                        f.unlink()
                        logger.info("Purged generated artifact: {}", f.name)

    return wrapper


def setup_dummy_sharegpt4v_coco(coco_dir: Path):
    """Enable ShareGPT4V to be used without downloading the actual COCO dataset."""
    coco_dir.mkdir(parents=True, exist_ok=True)

    dummy_image = Image.new("RGB", (256, 256))
    dummy_image_path = coco_dir / "dummy.png"
    dummy_image.save(dummy_image_path)

    raw_dataset, normalize_fn = load_raw_dataset("sharegpt4v_coco")

    # Use symlinks to avoid copying the image
    for raw_path in raw_dataset["image"]:
        image_path = coco_dir / raw_path.removeprefix("coco/")

        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.symlink_to(dummy_image_path)


def run_vllm_engine_and_assert(
    model_path: str,
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
    max_tokens: int = 50,
    acceptance_thresholds: Iterable[float] | None = None,
    **kwargs: Any,
) -> None:
    """Serve a checkpoint in vLLM, then assert on its outputs and acceptance rates.

    Extra keyword arguments are forwarded to
    :func:`scripts.pipeline_runners.run_vllm_engine`.
    """
    results_dict = pipeline_runners.run_vllm_engine(
        model_path=model_path,
        output_dir=tmp_path,
        prompts=prompts,
        max_tokens=max_tokens,
        **kwargs,
    )
    outputs_token_ids = results_dict["outputs"]
    metrics_dict = results_dict["metrics"]

    for output_token_ids in outputs_token_ids:
        # If max_tokens is 100 or less, make sure the output length is max_tokens
        assert max_tokens > 100 or len(output_token_ids) == max_tokens
        assert all(isinstance(token, int) for token in output_token_ids)

    if acceptance_thresholds is not None:
        for i, thresholdi in enumerate(acceptance_thresholds):
            assert f"acceptance_at_token_{i}" in metrics_dict, (
                f"Acceptance at token {i} is not in metrics_dict"
            )
            acci = metrics_dict[f"acceptance_at_token_{i}"]
            assert acci >= thresholdi, (
                f"Acceptance {acci} at token {i} is less than threshold {thresholdi}"
            )
