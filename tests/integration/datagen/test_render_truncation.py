"""Test that render endpoint truncation matches HF truncation.

Heavy/local test — spins up a real vLLM render server. Skips in CI
(requires vllm installed in a sibling venv).

Proves the render path is a drop-in replacement for the HF path:
same conversation + same seq_length → same truncated token_ids.
"""

import os
import shutil
import socket
import subprocess
import time

import pytest

try:
    import httpx
except ImportError:
    pytest.skip("httpx not available", allow_module_level=True)

from tests.e2e.utils import VLLM_PYTHON

MODEL = "Qwen/Qwen3-0.6B"
# Short enough to force truncation on a multi-turn conversation.
SEQ_LENGTH = 30


def _find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def render_server():
    if shutil.which(VLLM_PYTHON) is None:
        pytest.skip(f"vLLM python not found at {VLLM_PYTHON}")

    port = _find_free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            VLLM_PYTHON,
            "-m",
            "vllm.entrypoints.cli.main",
            "launch",
            "render",
            "--model",
            MODEL,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--max-model-len",
            "4096",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(1)
    else:
        proc.kill()
        pytest.skip("render server failed to start")

    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.mark.sanity
def test_render_truncation_matches_hf(render_server):
    """Truncated token_ids from the render endpoint must equal
    HF apply_chat_template with the same max_length."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(MODEL)

    conv = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the history of Paris."},
        {
            "role": "assistant",
            "content": "Paris has a long and fascinating history.",
        },
        {"role": "user", "content": "What about Rome?"},
        {"role": "assistant", "content": "Rome is the Eternal City."},
    ]

    # HF path: what speculators does today.
    full_ids = tok.apply_chat_template(
        conv,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    assert len(full_ids) > SEQ_LENGTH, "conversation too short to test truncation"
    hf_truncated = list(full_ids[:SEQ_LENGTH])

    # Render path: POST with truncate_prompt_tokens.
    resp = httpx.post(
        f"{render_server}/v1/chat/completions/render",
        json={
            "model": MODEL,
            "messages": conv,
            "add_generation_prompt": False,
            "truncate_prompt_tokens": SEQ_LENGTH,
            "truncation_side": "right",
        },
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    render_ids = resp.json()["token_ids"]

    assert render_ids == hf_truncated
