"""Capture FastMTP hidden states from a response-regenerated conversations file.

Takes a JSONL produced by scripts/response_regeneration/script.py (ShareGPT
conversations format), tokenizes it with the verifier's chat template and loss
mask, then runs a prefill-only vLLM pass to capture the last hidden layer.

Two-step workflow:
  Step 1 — regenerate responses (run once):
    scripts/response_regeneration/run_all.sh \\
        --model <verifier> --dataset gsm8k --tp-size 8

  Step 2 — capture hidden states (this script):
    python examples/fast_mtp/generate_dataset.py \\
        --model <verifier> \\
        --data-path gsm8k_Qwen3-Next-80B-A3B-Instruct.jsonl \\
        --output-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states

Output (one .pt file per sample):
    {
        "input_ids":     Tensor[seq_len],       # long
        "hidden_states": Tensor[seq_len, H],    # float32, last verifier layer
        "loss_mask":     Tensor[seq_len],       # long, 1 = assistant tokens only
    }
"""

import argparse
import logging
from pathlib import Path

from speculators.data_generation.fast_mtp_generator import generate_and_save_fast_mtp
from speculators.data_generation.preprocessing import load_and_preprocess_dataset

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture FastMTP hidden states from a conversations JSONL file"
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path or HF repo ID for the verifier model",
    )
    p.add_argument(
        "--data-path",
        required=True,
        help="JSONL file from scripts/response_regeneration/script.py "
        "(ShareGPT conversations format)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save .pt training files",
    )
    p.add_argument(
        "--symlink-dir",
        default="local/dataset",
        help="Relative path for a symlink pointing to output-dir "
        "(ignored when --no-symlink is set)",
    )
    p.add_argument(
        "--no-symlink",
        action="store_true",
        help="Skip creating a symlink to the output directory",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=8,
        help="CPU workers for tokenization",
    )
    return p.parse_args()


def ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target.resolve())
    print(f"Symlink: {link_path} -> {target.resolve()}")


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: tokenize conversations + build loss mask via existing pipeline
    log.info("Preprocessing %s ...", args.data_path)
    dataset, _ = load_and_preprocess_dataset(
        target_model_path=args.model,
        train_data_path=args.data_path,
        seq_length=args.max_model_len,
        build_dataset_num_proc=args.num_preprocessing_workers,
        max_samples=args.max_samples,
        token_freq_path=str(output_dir / "token_freq.pt"),
    )
    log.info("  %d samples after preprocessing", len(dataset))

    # Step 2: prefill-only vLLM pass to capture last hidden layer
    token_ids: list[list[int]] = dataset["input_ids"]
    loss_masks: list[list[int]] = dataset["loss_mask"]

    loss_mask_map = {
        tuple(ids): mask for ids, mask in zip(token_ids, loss_masks, strict=True)
    }

    _missing_mask_count = 0

    def loss_mask_fn(ids: list[int]) -> list[int]:
        nonlocal _missing_mask_count
        result = loss_mask_map.get(tuple(ids))
        if result is None:
            _missing_mask_count += 1
            log.warning(
                "No loss mask found for sequence (len=%d); defaulting to all-ones. "
                "Total missing so far: %d",
                len(ids),
                _missing_mask_count,
            )
            return [1] * len(ids)
        return result

    log.info("Capturing hidden states → %s ...", output_dir)
    generate_and_save_fast_mtp(
        model_path=args.model,
        token_ids=token_ids,
        output_dir=output_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        loss_mask_fn=loss_mask_fn,
    )

    n_files = len(list(output_dir.glob("data_*.pt")))
    log.info("Done. %d .pt files saved to %s", n_files, output_dir)

    if not args.no_symlink:
        repo_root = Path(__file__).resolve().parent.parent.parent
        symlink_path = repo_root / args.symlink_dir / output_dir.name
        ensure_symlink(output_dir, symlink_path)
        log.info("Symlink: %s", symlink_path)


if __name__ == "__main__":
    main()
