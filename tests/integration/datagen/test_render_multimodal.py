"""Faithful multimodal test for the render fallback — real Qwen3-VL render server.

Heavy/local test (skips in CI). Reproduces a real failure: vLLM does NOT truncate
multimodal prompts to ``truncate_prompt_tokens``, so for a conversation that exceeds
``seq_length`` the server returns more ``token_ids`` than ``max_length`` while the
local regex fallback truncates the mask to ``max_length`` -> ``input_ids`` and
``loss_mask`` end up different lengths.

Set ``RENDER_ENDPOINT`` to reuse an already-running Qwen3-VL render server; otherwise
one is launched from the sibling vLLM venv.
"""

import os
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

try:
    import httpx
except ImportError:
    pytest.skip("httpx not available", allow_module_level=True)

from datasets import Dataset as HFDataset
from PIL import Image

from speculators.data_generation.preprocessing import (
    build_dataset_from_render,
    load_processor,
)
from tests.e2e.utils import VLLM_PYTHON

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def test_dir(tmp_path_factory):
    """Module-scoped isolated temp dir: holds the generated image and doubles
    as the server's --allowed-local-media-path root."""
    return tmp_path_factory.mktemp("speculators_render")


@pytest.fixture(scope="module")
def image(test_dir):
    img_path = test_dir / "render_mm_test.png"
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :, 0] = 200
    arr[64:, :, 2] = 200
    arr[32:96, 32:96, 1] = 180
    Image.fromarray(arr).save(img_path)
    return str(img_path)


@pytest.fixture(scope="module")
def render_server(test_dir):
    env_endpoint = os.environ.get("RENDER_ENDPOINT")
    if env_endpoint:
        yield env_endpoint.rstrip("/")
        return

    if shutil.which(VLLM_PYTHON) is None:
        pytest.skip(f"vLLM python not found at {VLLM_PYTHON}")

    port = _find_free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            VLLM_PYTHON, "-m", "vllm.entrypoints.cli.main", "launch", "render",
            "--model", MODEL, "--port", str(port), "--host", "127.0.0.1",
            "--max-model-len", "4096", "--gpu-memory-utilization", "0.3",
            "--allowed-local-media-path", str(test_dir),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(2)
    else:
        proc.kill()
        pytest.skip("render server failed to start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.mark.sanity
def test_multimodal_over_length_not_misaligned(render_server, image):
    """A multimodal conversation that exceeds max_length must never be emitted as
    a misaligned (input_ids vs loss_mask) pair. vLLM won't truncate the image
    prompt, so the render path drops the row -- parity with the default pipeline,
    which drops over-length multimodal rows too. (Before the fix this emitted a
    86-vs-48 misaligned sample.)"""
    processor = load_processor(MODEL)
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the colors."},
                {"type": "image", "path": image},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Red, blue, green."}],
        },
    ]
    # max_length below the image's own token count: vLLM can't fit it.
    ds = build_dataset_from_render(
        HFDataset.from_dict({"conversations": [conv]}),
        render_server,
        processor,
        max_length=48,
    )

    for i in range(len(ds)):
        n_ids = len(ds[i]["input_ids"])
        n_mask = len(ds[i]["loss_mask"])
        assert n_ids == n_mask, (
            f"misaligned sample emitted: input_ids={n_ids} loss_mask={n_mask}"
        )
    # over-length multimodal row is dropped (parity with the default pipeline)
    assert len(ds) == 0
