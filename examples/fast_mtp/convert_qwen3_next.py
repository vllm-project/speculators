"""Extract the FastMTP head from a Qwen3-Next checkpoint.

FastMTP predicts multiple future tokens per step using a single shared
transformer layer applied recursively.  This script isolates that layer —
plus the model's token embeddings and LM head — and saves a compact,
self-contained Speculators checkpoint.

Usage
-----
Edit the constants below, then run::

    python examples/fast_mtp/convert_qwen3_next.py
"""

from speculators.convert import FastMTPConverter

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OUTPUT = "./Qwen3-Next-80B-A3B-Instruct_mtp_speculator"
NUM_STEPS = 3


if __name__ == "__main__":
    FastMTPConverter().convert(
        input_path=MODEL,
        output_path=OUTPUT,
        base_model=MODEL,
        num_speculative_steps=NUM_STEPS,
        validate=True,
    )
