#!/usr/bin/env python
"""Fire greedy prompts at the probe server and collect per-request
speculative-decoding acceptance metrics from the trained checkpoint."""

import json
import time
import urllib.request

URL = "http://localhost:8002/v1/completions"
MODEL = "/tmp/models/Qwen3.8-27B"

PROMPTS = [
    "Explain the theory of relativity:",
    "The process of photosynthesis converts",
    "Write a Python function that sorts a list:",
    "The French Revolution began in",
    "Water boils at a temperature of",
    "In quantum mechanics, the uncertainty principle states that",
    "The largest planet in the solar system is",
    "A binary search algorithm works by",
    "def fibonacci(n):",
    "The capital of Australia is",
    "Machine learning is a field of",
    "Translate to French: The weather is beautiful today.",
    "SELECT * FROM users",
    "To make a cup of coffee, first",
    "The Pythagorean theorem states",
    "History of the Roman Empire:",
]

results = []
for p in PROMPTS:
    body = json.dumps(
        {"model": MODEL, "prompt": p, "max_tokens": 200, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    dt = time.time() - t0
    usage = resp.get("usage", {})
    m = (resp.get("metrics") or {}).get("speculative_decoding") or {}
    results.append({"prompt": p, "wall_s": round(dt, 2), "usage": usage, "spec": m})
    print(
        f"{p[:38]:38s} ctok={usage.get('completion_tokens')} wall={dt:.1f}s spec={m}",
        flush=True,
    )

with open(
    "/tmp/lilicorr_check/qwen38-perfectblend/runfull/acceptance_probe.json", "w"
) as f:
    json.dump(results, f, indent=1)

als = [r["spec"]["mean_acceptance_length"] for r in results if r["spec"]]
if als:
    print(
        f"\nper-request mean_acceptance_length: n={len(als)} "
        f"mean={sum(als) / len(als):.3f} min={min(als):.2f} max={max(als):.2f}"
    )
print("saved to runfull/acceptance_probe.json")
