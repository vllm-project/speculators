"""Generate FastMTP hidden state training data from GSM8K using Qwen3-Next.

Two-pass pipeline:
  1. Generate responses for each GSM8K question using vLLM autoregressive decoding.
  2. Run the complete (question + generated response) sequences through vLLM
     prefill-only to capture last-layer hidden states.

Output (one .pt file per sample):
    {
        "input_ids":     Tensor[seq_len],       # long
        "hidden_states": Tensor[seq_len, H],    # bfloat16, last verifier layer
        "loss_mask":     Tensor[seq_len],       # long, 1 = assistant tokens only
    }

Usage:
    python examples/fast_mtp/generate_dataset.py \\
        --model /path/to/Qwen3-Next-80B-A3B-Instruct \\
        --output-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \\
        --tensor-parallel-size 8 \\
        --max-new-tokens 2048 \\
        --max-model-len 4096
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, then give the final answer."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate FastMTP GSM8K hidden state dataset"
    )
    p.add_argument(
        "--model",
        default="/mnt/data/engine/hub_cache/models--Qwen--Qwen3-Next-80B-A3B-Instruct"
        "/snapshots/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7",
        help="Path or HF repo ID for the verifier model",
    )
    p.add_argument(
        "--output-dir",
        default="/mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states",
        help="Directory to save .pt training files",
    )
    p.add_argument(
        "--symlink-dir",
        default="local/dataset",
        help="Relative path for symlink pointing to output-dir",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for generation",
    )
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of GSM8K samples (for testing)",
    )
    p.add_argument(
        "--generation-cache",
        default=None,
        help="Path to save/load generated sequences (avoids re-running generation)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Build prompts from GSM8K
# ---------------------------------------------------------------------------


def build_prompts(tokenizer: AutoTokenizer, limit: int | None = None) -> list[dict]:
    """Load GSM8K train split and format each question as a chat prompt.

    Returns list of dicts with keys: prompt_text, prompt_ids, answer_text.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    samples = []
    for item in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        # apply_chat_template with add_generation_prompt=True gives the model
        # the open-ended prompt it should complete.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        samples.append(
            {
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "reference_answer": item["answer"],
            }
        )
    print(f"Loaded {len(samples)} GSM8K train samples")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Generate responses with vLLM
# ---------------------------------------------------------------------------


def generate_responses(
    model_path: str,
    samples: list[dict],
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
) -> list[dict]:
    """Run vLLM generation to produce a response for each sample.

    Returns samples augmented with full_ids (prompt + response token IDs)
    and loss_mask (1 for response tokens, 0 for prompt tokens).
    """
    from vllm import LLM, SamplingParams

    print(f"[Pass 1] Generating responses with TP={tensor_parallel_size} ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    prompts = [s["prompt_text"] for s in samples]
    outputs = llm.generate(prompts, sampling_params)

    # Free GPU memory before pass 2
    del llm
    torch.cuda.empty_cache()

    results = []
    for sample, output in zip(samples, outputs):
        prompt_ids = sample["prompt_ids"]
        generated_ids = list(output.outputs[0].token_ids)
        full_ids = prompt_ids + generated_ids

        # Loss mask: 0 for prompt tokens, 1 for generated (assistant) tokens
        loss_mask = [0] * len(prompt_ids) + [1] * len(generated_ids)

        results.append(
            {
                "full_ids": full_ids,
                "loss_mask": loss_mask,
                "reference_answer": sample["reference_answer"],
                "prompt_len": len(prompt_ids),
                "response_len": len(generated_ids),
            }
        )

    n_truncated = sum(1 for r in results if r["response_len"] == max_new_tokens)
    if n_truncated:
        print(f"  Warning: {n_truncated} responses hit max_new_tokens={max_new_tokens}")
    print(f"  Generated {len(results)} responses")
    return results


# ---------------------------------------------------------------------------
# Step 3: Capture hidden states with vLLM prefill
# ---------------------------------------------------------------------------


def capture_hidden_states(
    model_path: str,
    sequences: list[dict],
    output_dir: Path,
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> None:
    """Run prefill-only pass to capture last-layer hidden states and save .pt files."""
    from speculators.data_generation.fast_mtp_generator import (
        generate_and_save_fast_mtp,
    )

    token_ids = [s["full_ids"] for s in sequences]

    # Build a loss_mask_fn closure using the pre-computed masks.
    # The generator calls loss_mask_fn(token_ids) for each sequence; we index
    # by position in the list using a precomputed mapping.
    id_to_mask = {tuple(s["full_ids"]): s["loss_mask"] for s in sequences}

    def loss_mask_fn(ids: list[int]) -> list[int]:
        key = tuple(ids)
        if key in id_to_mask:
            return id_to_mask[key]
        # Fallback: only last half of sequence (should not happen)
        return [0] * (len(ids) // 2) + [1] * (len(ids) - len(ids) // 2)

    print(f"[Pass 2] Capturing hidden states with TP={tensor_parallel_size} ...")
    generate_and_save_fast_mtp(
        model_path=model_path,
        token_ids=token_ids,
        output_dir=output_dir,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        loss_mask_fn=loss_mask_fn,
    )


# ---------------------------------------------------------------------------
# Symlink helper
# ---------------------------------------------------------------------------


def ensure_symlink(target: Path, link_path: Path) -> None:
    """Create or update a symlink at link_path pointing to target."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target.resolve())
    print(f"Symlink: {link_path} -> {target.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config for reproducibility
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Load tokenizer (needed to build prompts)
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build prompts from GSM8K
    samples = build_prompts(tokenizer, limit=args.limit)

    # Step 1: generate or load cached responses
    generation_cache = (
        Path(args.generation_cache)
        if args.generation_cache
        else output_dir / "generations.json"
    )
    if generation_cache.exists():
        print(f"Loading cached generations from {generation_cache}")
        with generation_cache.open() as f:
            sequences = json.load(f)
        print(f"  Loaded {len(sequences)} cached sequences")
    else:
        sequences = generate_responses(
            model_path=args.model,
            samples=samples,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        with generation_cache.open("w") as f:
            json.dump(sequences, f)
        print(f"Saved generations to {generation_cache}")

    # Filter sequences that are too long for the hidden state extractor
    before = len(sequences)
    sequences = [s for s in sequences if len(s["full_ids"]) <= args.max_model_len]
    if len(sequences) < before:
        n_filtered = before - len(sequences)
        print(f"  Filtered {n_filtered} sequences exceeding max_model_len")

    # Step 2: capture hidden states and save .pt files
    capture_hidden_states(
        model_path=args.model,
        sequences=sequences,
        output_dir=output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Create symlink in local/dataset/
    repo_root = Path(__file__).resolve().parent.parent.parent
    symlink_path = repo_root / args.symlink_dir / output_dir.name
    ensure_symlink(output_dir, symlink_path)

    print(f"\nDone. Dataset saved to: {output_dir}")
    print(f"  Symlink at: {symlink_path}")
    print(f"  Files: {len(list(output_dir.glob('data_*.pt')))} .pt samples")


if __name__ == "__main__":
    main()
