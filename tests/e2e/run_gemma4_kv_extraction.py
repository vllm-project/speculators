"""Subprocess driver for the Gemma4KVConnector extraction test.

Generic, assertion-free counterpart to run_vllm.py: it is parametrized by
JSON (--llm-args, --prompts) and dumps raw per-request facts (tensor
shapes, token alignment, finiteness) to --results-file. All assertions live
parent-side in tests/e2e/utils.run_gemma4_kv_extraction.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# isort: off
from tests.e2e.run_vllm import LLM, SamplingParams  # noqa: E402

import torch  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    example_hidden_states_connector,
)

# isort: on

from speculators.data_generation.gemma4_kv_connector import (  # noqa: E402
    GEMMA4_KV_KEYS as KV_KEYS,
)


def run_extraction(llm_args: dict, prompts: list[str]) -> dict:
    """Run the extraction forward pass and collect raw per-request facts.

    No assertions: this only reports JSON-serializable facts (shapes, token
    alignment, finiteness) for the parent process to assert on.
    """
    llm = LLM(**llm_args)

    model_config = llm.llm_engine.model_config
    hidden_size = model_config.get_hidden_size()
    text_config = model_config.hf_config.get_text_config()
    local_total_heads = text_config.num_key_value_heads
    global_total_heads = (
        getattr(text_config, "num_global_key_value_heads", None) or local_total_heads
    )

    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0))

    per_request: list[dict[str, Any]] = []
    for output in outputs:
        kv_params = output.kv_transfer_params
        path = kv_params.get("hidden_states_path") if kv_params else None
        if path is None:
            per_request.append({"hidden_states_path": None})
            continue

        obj = example_hidden_states_connector.load_hidden_states(path)
        n = len(output.prompt_token_ids)
        per_request.append(
            {
                "hidden_states_path": path,
                "num_tokens": n,
                "token_ids_aligned": bool(
                    torch.equal(obj["token_ids"], torch.tensor(output.prompt_token_ids))
                ),
                "hidden_states_shape": list(obj["hidden_states"].shape),
                "kv": {
                    key: {
                        "present": key in obj,
                        "shape": list(obj[key].shape) if key in obj else None,
                        "finite": bool(torch.isfinite(obj[key]).all())
                        if key in obj
                        else None,
                    }
                    for key in KV_KEYS
                },
            }
        )

    return {
        "num_outputs": len(outputs),
        "hidden_size": hidden_size,
        "local_total_heads": local_total_heads,
        "global_total_heads": global_total_heads,
        "per_request": per_request,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-args",
        type=str,
        required=True,
        help="JSON-serialized kwargs for LLM instantiation",
    )
    parser.add_argument(
        "--prompts", type=str, required=True, help="JSON-serialized list of prompts"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="File to write the JSON-serialized extraction facts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        results = run_extraction(json.loads(args.llm_args), json.loads(args.prompts))
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        results = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1

    with args.results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    sys.exit(exit_code)
