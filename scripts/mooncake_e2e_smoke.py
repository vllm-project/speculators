"""End-to-end smoke test for the Mooncake hidden-states backend.

Exercises the full seam against *live* services: send a completion to a vLLM
target running the Mooncake connector, get a ``mooncake_key`` back, then read
the extracted hidden states out of the Mooncake store from a separate client
(standing in for a trainer on another node).

Prerequisites:
  1. A Mooncake master:   mooncake_master --port 50051
  2. A vLLM target:       python scripts/launch_vllm.py <MODEL> \
                              --hidden-states-backend mooncake \
                              --mooncake-master 127.0.0.1:50051 \
                              --mooncake-metadata-server P2PHANDSHAKE -- --port 8000
  (mooncake needs libcudart on LD_LIBRARY_PATH; on a torch-cu13 venv, point it
   at $VENV/lib/python3.12/site-packages/nvidia/cu13/lib)

Usage:
  python scripts/mooncake_e2e_smoke.py --model <MODEL> [--num-aux 4] [--hidden 2048]
"""

import argparse
import json
import urllib.request

from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
    resolve_local_hostname,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--master", default="127.0.0.1:50051")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--num-aux", type=int, default=4, help="len(target_layer_ids)")
    ap.add_argument("--hidden", type=int, default=2048, help="model hidden size")
    ap.add_argument(
        "--local-hostname",
        default=None,
        help="This client's routable address for P2P handshake (cross-node "
        "reads). Default: auto-resolved from the route to the master.",
    )
    ap.add_argument("--protocol", default="tcp", choices=["tcp", "rdma"])
    ap.add_argument("--device", default="", help="RDMA device for --protocol rdma")
    args = ap.parse_args()

    # 1) Producer: vLLM extracts hidden states into Mooncake, returns the key.
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": 1,
        "return_token_ids": True,
    }
    req = urllib.request.Request(  # noqa: S310 (localhost smoke test)
        f"{args.endpoint}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())  # noqa: S310
    key = resp["kv_transfer_params"]["mooncake_key"]
    ptids = resp["choices"][0].get("prompt_token_ids") or resp.get("prompt_token_ids")
    print(f"mooncake_key={key}  num_prompt_tokens={len(ptids)}")

    # 2) Consumer (trainer): read the sample back over the network by key.
    store = MooncakeHiddenStatesStore(
        MooncakeStoreConfig.for_consumer(
            local_hostname=args.local_hostname or resolve_local_hostname(args.master),
            metadata_server="P2PHANDSHAKE",
            master_server_address=args.master,
            protocol=args.protocol,
            device_name=args.device,
        )
    ).setup()
    out = store.get_sample(key, timeout=30.0)

    hs, ids = out["hidden_states"], out["token_ids"]
    print(f"hidden_states={tuple(hs.shape)} {hs.dtype}  token_ids={tuple(ids.shape)}")

    ok = (
        list(hs.shape) == [len(ids), args.num_aux, args.hidden]
        and ids.tolist() == ptids[: len(ids)]
    )
    print("E2E FULL-LOOP", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
