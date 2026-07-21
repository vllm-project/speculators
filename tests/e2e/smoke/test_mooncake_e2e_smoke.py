"""E2E smoke test for the Mooncake hidden-states producer/consumer loop.

Sends a single completion to a vLLM server running the Mooncake connector,
retrieves the ``mooncake_key`` from the response, then reads the hidden states
back from a separate MooncakeHiddenStatesStore client (standing in for a
trainer on another node).  Validates shape and token-id alignment.

Prerequisites are launched automatically (mooncake_master + vLLM server).
The test is skipped when the ``hs_connectors`` extra or the ``mooncake_master``
binary are not installed.
"""

import json
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.utils import launch_mooncake_master_context, launch_vllm_server_context

mc_store = pytest.importorskip(
    "hs_connectors.mooncake_store",
    reason="hs_connectors[mooncake] not installed",
)
MooncakeHiddenStatesStore = mc_store.MooncakeHiddenStatesStore
MooncakeStoreConfig = mc_store.MooncakeStoreConfig

MODEL = "Qwen/Qwen3-0.6B"
MOONCAKE_MASTER_PORT = 50052
MOONCAKE_MASTER_ADDR = f"127.0.0.1:{MOONCAKE_MASTER_PORT}"
VLLM_PORT = 8323


def _send_completion(endpoint: str, model: str, prompt: str) -> dict:
    """Send a single /v1/completions request and return the parsed response."""
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
def test_mooncake_hidden_states_roundtrip(tmp_path: Path):
    """Producer (vLLM) writes hidden states via Mooncake; consumer reads them back."""
    prompt = "The capital of France is"

    with (
        launch_mooncake_master_context(MOONCAKE_MASTER_PORT),
        launch_vllm_server_context(
            MODEL,
            VLLM_PORT,
            hidden_states_path=str(tmp_path / "hidden_states"),
            hidden_states_backend="mooncake",
            mooncake_master=MOONCAKE_MASTER_ADDR,
            mooncake_metadata_server="P2PHANDSHAKE",
            mooncake_protocol="tcp",
            enforce_eager=True,
        ),
    ):
        resp = _send_completion(f"http://127.0.0.1:{VLLM_PORT}", MODEL, prompt)

        key = resp["kv_transfer_params"]["mooncake_key"]
        ptids = resp["choices"][0].get("prompt_token_ids") or resp.get(
            "prompt_token_ids"
        )
        assert key, "mooncake_key missing from response"
        assert ptids, "prompt_token_ids missing from response"

        store = MooncakeHiddenStatesStore(
            MooncakeStoreConfig(
                local_hostname="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                master_server_address=MOONCAKE_MASTER_ADDR,
                protocol="tcp",
                global_segment_size=256 * 1024 * 1024,
                local_buffer_size=128 * 1024 * 1024,
            )
        ).setup()

        out = store.get_sample(key, timeout=30.0)
        hs, ids = out["hidden_states"], out["token_ids"]

        assert hs.ndim == 3, f"expected 3-d hidden_states, got shape {hs.shape}"
        assert hs.shape[0] == len(ids), (
            f"seq dim mismatch: hidden_states {hs.shape[0]} vs token_ids {len(ids)}"
        )
        assert ids.tolist() == ptids[: len(ids)]
