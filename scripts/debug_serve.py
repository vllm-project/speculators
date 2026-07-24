"""Wrap vllm serve with faulthandler to catch native crashes.

Usage (from the speculators dir, with spec env active):
    CUDA_VISIBLE_DEVICES=0 python scripts/debug_serve.py Qwen/Qwen3-8B \
        --target-layer-ids 2 18 33
"""

import json
import os
import subprocess
import sys
import warnings


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[2, 18, 33])
    parser.add_argument("--port", type=int, default=8000)
    args, extra = parser.parse_known_args()

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model)
    if hasattr(config, "text_config"):
        config = config.text_config
    num_hidden_layers = config.num_hidden_layers

    target_layer_ids = list(args.target_layer_ids)
    if num_hidden_layers not in target_layer_ids:
        target_layer_ids.append(num_hidden_layers)

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": "/tmp/hidden_states",
        },
    }

    cmd = [
        sys.executable,
        "-Xfaulthandler",
        "-m", "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config", json.dumps(speculative_config),
        "--kv_transfer_config", json.dumps(kv_transfer_config),
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "--port", str(args.port),
        *extra,
    ]

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    print("Running:", " ".join(cmd), flush=True)
    print("With PYTHONFAULTHANDLER=1 and CUDA_LAUNCH_BLOCKING=1", flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
