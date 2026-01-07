#!/usr/bin/env python3
"""
Regenerate responses from Magpie instructions using vLLM offline inference.

This script:
1. Loads instructions from a dataset
2. Generates responses using a vLLM model (supports reasoning/thinking models)
3. Tokenizes with chat templates
4. Creates loss masks (0=user tokens, 1=assistant tokens including thinking)
5. Outputs JSONL with input_ids and loss_mask for training

Output format is compatible with data_generation_offline.py for EAGLE training.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Set

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

from speculators.data_generation.preprocessing import _visualize_sample

DATASET_ID = "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered"

# Set up logger
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate responses from Magpie instructions via vLLM offline inference."
    )
    p.add_argument(
        "--model", default="openai/gpt-oss-20b", help="Model name to load"
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    p.add_argument(
        "--batch-size", type=int, default=256, help="Number of prompts per batch"
    )
    p.add_argument(
        "--max-tokens", type=int, default=8192, help="max_tokens for generation"
    )
    p.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU",
    )
    p.add_argument(
        "--outfile",
        default="magpie_responses.jsonl",
        help="Where to write JSONL results",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already in outfile (by uuid or idx)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize loss masks for first batch (debugging)",
    )
    return p.parse_args()


def load_seen(path: str) -> Set[str]:
    """Load already processed UUIDs from output file for resume capability."""
    seen = set()
    if not os.path.isfile(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                uuid = obj.get("uuid")
                if uuid is not None:
                    seen.add(str(uuid))
            except Exception:
                continue
    return seen


def process_batch(llm, sampling_params, prompts, metadata, out_fh, args, visualize_batch=False):
    """Generate responses for a batch of prompts and write results.

    For each prompt:
    1. Generate response (may include thinking tokens)
    2. Apply chat template
    3. Tokenize
    4. Create loss mask: 0 for user, 1 for all assistant content
    5. Write to output file
    """
    logger.info(f"Generating {len(prompts)} responses...")
    t0 = time.time()

    try:
        # Generate responses
        outputs = llm.generate(prompts, sampling_params)

        batch_time = time.time() - t0
        throughput = len(prompts) / batch_time
        logger.info(
            f"Batch completed in {batch_time:.2f}s ({throughput:.1f} req/s)"
        )

        tokenizer = llm.get_tokenizer()
        results = []

        for output, meta in zip(outputs, metadata):
            generated_text = output.outputs[0].text

            # Create conversation (includes all assistant content: thinking + answer)
            conversation = [
                {"role": "user", "content": meta["instruction"]},
                {"role": "assistant", "content": generated_text},
            ]

            # Tokenize full conversation with chat template
            formatted_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            encoding = tokenizer(formatted_text, add_special_tokens=False)
            input_ids = encoding["input_ids"]

            # Tokenize user-only to find assistant boundary
            user_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": meta["instruction"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            user_encoding = tokenizer(user_text, add_special_tokens=False)
            user_len = len(user_encoding["input_ids"])

            # Create loss mask: 0 for user, 1 for assistant (including thinking)
            loss_mask = [0] * user_len + [1] * (len(input_ids) - user_len)

            # Output in conversations format for data_generation_offline.py
            result = {
                "conversations": [
                    {"role": "user", "content": meta["instruction"]},
                    {"role": "assistant", "content": generated_text},
                ],
                "uuid": meta.get("uuid"),  # For resume capability
            }
            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Collect for visualization if needed
            if visualize_batch:
                results.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
                })

        out_fh.flush()

        # Visualize first sample if requested
        if visualize_batch and results:
            _visualize_sample(None, results, tokenizer, idx=0)

    except Exception as e:
        logger.error(
            f"Error processing batch (discarding {len(metadata)} items): {e}",
            exc_info=True
        )


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure preprocessing module's logger for visualization
    if args.visualize:
        preprocessing_logger = logging.getLogger('speculators.data_generation.preprocessing')
        preprocessing_logger.setLevel(logging.INFO)

    # Load already processed items if resuming
    seen = load_seen(args.outfile) if args.resume else set()
    if seen:
        logger.info(f"Resuming: skipping {len(seen)} already processed items")

    # Initialize vLLM model
    logger.info(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Load dataset in streaming mode
    logger.info(f"Loading dataset {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    # Process dataset in batches
    with open(args.outfile, "a", encoding="utf-8") as out_fh:
        batch = []
        batch_metadata = []
        processed = 0
        batch_count = 0

        for i, row in enumerate(ds):
            if args.limit is not None and processed >= args.limit:
                break

            # Skip rows without instructions
            instr = row.get("instruction")
            if not instr:
                continue

            # Check if already processed (for resume)
            uid = row.get("uuid")
            if args.resume and uid and str(uid) in seen:
                continue

            # Add to batch
            batch.append(instr)
            batch_metadata.append({"uuid": uid, "instruction": instr})
            processed += 1

            # Process when batch is full
            if len(batch) >= args.batch_size:
                # Only visualize first batch
                visualize = args.visualize and batch_count == 0
                process_batch(llm, sampling_params, batch, batch_metadata, out_fh, args, visualize)
                batch = []
                batch_metadata = []
                batch_count += 1

        # Process remaining items
        if batch:
            visualize = args.visualize and batch_count == 0
            process_batch(llm, sampling_params, batch, batch_metadata, out_fh, args, visualize)

    logger.info(f"Completed! Processed {processed} items total.")
    logger.info(f"Output written to: {args.outfile}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)  # Ensure errors always show
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
