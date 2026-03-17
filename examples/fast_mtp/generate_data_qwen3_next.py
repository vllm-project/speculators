"""Generate FastMTP training data from Qwen3-Next-80B-A3B-Instruct.

Runs prefill on Qwen3-Next-80B-A3B-Instruct, captures the last verifier hidden
state for each token, and saves one ``.pt`` file per sequence to OUTPUT_DIR.

Each file contains::

    {
        "input_ids":     Tensor[seq_len],        # long
        "hidden_states": Tensor[seq_len, H],     # float32, last verifier layer
        "loss_mask":     Tensor[seq_len],        # 1 = train on this token
    }

Usage
-----
Edit the constants below, then run::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/fast_mtp/generate_data_qwen3_next.py
"""

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from speculators.data_generation.fast_mtp_generator import generate_and_save_fast_mtp
from speculators.data_generation.preprocessing import tokenize_conversations

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OUTPUT_DIR = Path("./output/Qwen3-Next-80B-A3B-Instruct_ultrachat")
DATASET = "HuggingFaceH4/ultrachat_200k"
NUM_SAMPLES = 100
MAX_SEQ_LEN = 4096
TENSOR_PARALLEL_SIZE = 4  # match the number of GPUs in CUDA_VISIBLE_DEVICES
GPU_MEMORY_UTILIZATION = 0.85

# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # ── Tokenize ──────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {DATASET} (train_sft split, {NUM_SAMPLES} samples)")
    raw = load_dataset(DATASET, split="train_sft")
    raw = raw.shuffle(seed=42).select(range(NUM_SAMPLES))
    # ultrachat_200k stores turns under "messages"; tokenize_conversations expects
    # the standard "conversations" key used by ShareGPT-format datasets.
    raw = raw.map(
        lambda ex: {"conversations": ex["messages"]},
        remove_columns=raw.column_names,
    )

    print(f"Tokenizing {NUM_SAMPLES} samples (max_len={MAX_SEQ_LEN})")
    dataset = tokenize_conversations(
        dataset=raw,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        num_proc=4,
    )

    token_ids = [s["input_ids"].tolist() for s in dataset]
    loss_masks = [s["loss_mask"].tolist() for s in dataset]

    lengths = [len(t) for t in token_ids]
    print(
        f"Sequence lengths — min: {min(lengths)}, "
        f"max: {max(lengths)}, "
        f"mean: {sum(lengths) / len(lengths):.0f}"
    )

    # ── Generate and save ─────────────────────────────────────────────────────
    print(f"Generating hidden states and saving to {OUTPUT_DIR}")
    generate_and_save_fast_mtp(
        model_path=MODEL,
        token_ids=token_ids,
        loss_masks=loss_masks,
        output_dir=OUTPUT_DIR,
        max_model_len=MAX_SEQ_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )

    print(f"\nDone. {NUM_SAMPLES} samples saved to {OUTPUT_DIR}")
