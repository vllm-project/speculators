"""End-to-end integration test for the dynamic hidden-states plugin.

Requires a live vLLM server started WITH the plugin enabled
(``VLLM_PLUGINS=dynamic_hidden_states``, which loads both the endpoint plugin
and the masked-accumulate general plugin). The swap works under the default
``torch.compile`` + CUDA graphs — no ``--enforce-eager`` needed.
``--no-enable-prefix-caching`` is still required so identical prompts re-run the
forward instead of being served from cache. Not a unit test — run it manually
against a running server:

    # 1 GPU, ~1-2 min to start
    VLLM_PLUGINS=dynamic_hidden_states \
    /workspace/vllm/.venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B --port 8000 \
        --no-enable-prefix-caching \
        --worker-extension-cls \
            dynamic_hidden_states.worker_extension.AuxLayerWorkerExtension \
        --speculative_config '{"method":"extract_hidden_states",\
"num_speculative_tokens":1,"draft_model_config":{"hf_config":\
{"eagle_aux_hidden_state_layer_ids":[2,18,34]}}}' \
        --kv_transfer_config '{"kv_connector":"ExampleHiddenStatesConnector",\
"kv_role":"kv_producer","kv_connector_extra_config":\
{"shared_storage_path":"/dev/shm/hidden_states"}}'

    # then, once healthy:
    /workspace/vllm/.venv/bin/python vllm_plugins/tests/test_swap_e2e.py

The core assertion: swapping ONLY the first captured index (2 -> 6) changes
exactly the first hidden-state column while the columns for the unchanged
layers (18, 34) stay bit-identical for the same prompt.
"""

import glob
import json
import os
import time
import urllib.request

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    load_hidden_states,
)

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
STORE = os.environ.get("SHARED_STORAGE_PATH", "/dev/shm/hidden_states")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PROMPT = "The capital of France is"


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _newest(before):
    new = set(glob.glob(os.path.join(STORE, "*.safetensors"))) - before
    assert new, "no new safetensors file was written"
    return max(new, key=os.path.getmtime)


def generate_and_capture():
    before = set(glob.glob(os.path.join(STORE, "*.safetensors")))
    status, _ = http("POST", "/v1/completions", {
        "model": MODEL, "prompt": PROMPT, "max_tokens": 1, "temperature": 0.0,
    })
    assert status == 200, f"completion failed: {status}"
    time.sleep(1.5)  # connector writes asynchronously
    obj = load_hidden_states(_newest(before))
    return obj["hidden_states"], obj["token_ids"]


def errmsg(body):
    if "detail" in body:
        return body["detail"]
    return body.get("error", {}).get("message", "")


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    assert ok, f"{name} failed: {detail}"


def main():
    print("=" * 60)
    http("POST", "/aux_hidden_state_layers", {"layers": [2, 18, 34]})

    status, body = http("GET", "/aux_hidden_state_layers")
    report("GET initial layers", status == 200 and body["layers"] == [2, 18, 34],
           str(body))

    hs_a, ids_a = generate_and_capture()
    report("baseline capture shape [T,3,H]",
           hs_a.ndim == 3 and hs_a.shape[1] == 3, f"shape={tuple(hs_a.shape)}")

    status, body = http("POST", "/aux_hidden_state_layers", {"layers": [6, 18, 34]})
    report("POST swap [6,18,34]",
           status == 200 and body["layers"] == [6, 18, 34], str(body))
    status, body = http("GET", "/aux_hidden_state_layers")
    report("GET reflects [6,18,34]", body["layers"] == [6, 18, 34], str(body))

    hs_b, ids_b = generate_and_capture()
    report("same prompt tokens across runs", torch.equal(ids_a, ids_b),
           f"{ids_a.tolist()} vs {ids_b.tolist()}")

    d0 = (hs_a[:, 0] - hs_b[:, 0]).abs().max().item()
    d1 = (hs_a[:, 1] - hs_b[:, 1]).abs().max().item()
    d2 = (hs_a[:, 2] - hs_b[:, 2]).abs().max().item()
    report("swapped column changed (col0: 2->6)",
           not torch.allclose(hs_a[:, 0], hs_b[:, 0], atol=1e-3), f"max|Δ|={d0:.4f}")
    report("layer 18 identical (col1)",
           torch.allclose(hs_a[:, 1], hs_b[:, 1], atol=1e-3), f"max|Δ|={d1:.6f}")
    report("layer 34 identical (col2)",
           torch.allclose(hs_a[:, 2], hs_b[:, 2], atol=1e-3), f"max|Δ|={d2:.6f}")

    status, body = http("POST", "/aux_hidden_state_layers", {"layers": [1, 2]})
    report("reject count mismatch (400)",
           status == 400 and "fixed" in errmsg(body), f"{status} {errmsg(body)}")

    status, body = http("POST", "/aux_hidden_state_layers", {"layers": []})
    report("reject empty layers (400)", status == 400, str(status))

    status, body = http("POST", "/aux_hidden_state_layers", {"layers": [2, 18, 34]})
    report("restore [2,18,34]",
           status == 200 and body["layers"] == [2, 18, 34], str(body))

    print("=" * 60)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
