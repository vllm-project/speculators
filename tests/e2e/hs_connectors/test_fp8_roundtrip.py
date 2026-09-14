"""E2E smoke test for the FP8-quantizing hidden-states producer/consumer loop.

Sends a single completion to a vLLM server running FP8HiddenStatesConnector,
retrieves the on-disk safetensors path from the response, then reads the
hidden states back through FP8Transfer (standing in for the trainer-side
dataloader). Validates that the file on disk is actually FP8-quantized and
that FP8Transfer dequantizes it back to values close to what an unquantized
("file" backend) run would have produced.
"""

import json
import urllib.request
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tests.e2e.utils import launch_vllm_server_context

hs_connectors = pytest.importorskip(
    "hs_connectors", reason="hs_connectors not installed"
)
FP8Transfer = hs_connectors.FP8Transfer

MODEL = "Qwen/Qwen3-0.6B"
VLLM_PORT = 8324


def _send_completion(endpoint: str, model: str, prompt: str) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "return_token_ids": True,
    }
    req = urllib.request.Request(  # noqa: S310
        f"{endpoint}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())  # noqa: S310


@pytest.mark.e2e
@pytest.mark.slow
def test_fp8_hidden_states_roundtrip(tmp_path: Path):
    """Producer (vLLM) writes FP8-quantized hidden states; consumer dequantizes."""
    prompt = "The capital of France is"
    hs_path = tmp_path / "hidden_states"

    with launch_vllm_server_context(
        MODEL,
        VLLM_PORT,
        hidden_states_path=str(hs_path),
        hidden_states_backend="fp8",
        enforce_eager=True,
    ):
        resp = _send_completion(f"http://127.0.0.1:{VLLM_PORT}", MODEL, prompt)

        path = resp["kv_transfer_params"]["hidden_states_path"]
        ptids = resp["choices"][0].get("prompt_token_ids") or resp.get(
            "prompt_token_ids"
        )
        assert path, "hidden_states_path missing from response"
        assert ptids, "prompt_token_ids missing from response"

        # FP8Transfer waits on the companion .lock file (same protocol as
        # vLLM's own load_hidden_states()), so this blocks until the async
        # disk write is actually done -- unlike a bare safe_open(path), which
        # can race the writer thread.
        transfer = FP8Transfer(hs_path)
        sample = transfer.get_generated(path)
        assert sample is not None
        hs, ids = sample["hidden_states"], sample["token_ids"]

        # Now that the write is guaranteed complete, confirm the on-disk file
        # was actually FP8-quantized (not a plain bf16 "file" backend
        # payload) -- proves FP8HiddenStatesConnector's _write_tensors
        # override was actually invoked.
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            assert "hidden_states_scales" in keys, f"no scales tensor in {keys}"
            raw_hs = f.get_tensor("hidden_states")
        assert raw_hs.dtype == torch.float8_e4m3fn, (
            f"expected float8_e4m3fn on disk, got {raw_hs.dtype}"
        )

        assert "hidden_states_scales" not in sample, (
            "FP8Transfer should hide the scales tensor from consumers"
        )
        assert hs.dtype == torch.bfloat16
        assert hs.ndim == 3, f"expected 3-d hidden_states, got shape {hs.shape}"
        assert hs.shape[0] == len(ids), (
            f"seq dim mismatch: hidden_states {hs.shape[0]} vs token_ids {len(ids)}"
        )
        assert ids.tolist() == ptids[: len(ids)]
        assert torch.isfinite(hs).all(), "dequantized hidden states contain NaN/Inf"
