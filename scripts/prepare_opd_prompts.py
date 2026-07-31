"""Prepare the Draft-OPD prompt pool (16K prompts).

Downloads and samples from the four sources used in the paper:
  - 2K prompts from GSM8K training set
  - 5K prompts from MATH corpus (excluding MATH-500 held-out)
  - 4K prompts from AoPS (NuminaMath-CoT)
  - 5K prompts from CodeAlpaca

Tokenizes each prompt with the target model's chat template and saves
as a jsonl file with one {"input_ids": [...], "source": "..."} per line.

Usage:
    python scripts/prepare_opd_prompts.py \
        --model Qwen/Qwen3-8B \
        --output opd_prompts.jsonl
"""

import argparse
import json
import logging
import random

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def sample_gsm8k(n: int = 2000) -> list[dict]:
    logger.info("Loading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"prompt": ds[i]["question"], "source": "gsm8k"} for i in indices]


def sample_math(n: int = 5000) -> list[dict]:
    logger.info("Loading MATH dataset...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"prompt": ds[i]["problem"], "source": "math"} for i in indices]


def sample_aops(n: int = 4000) -> list[dict]:
    logger.info("Loading AoPS from NuminaMath-CoT (aops_forum subset)...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    ds = ds.filter(lambda x: x["source"] == "aops_forum")
    logger.info("  aops_forum subset: %d examples", len(ds))
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"prompt": ds[i]["problem"], "source": "aops"} for i in indices]


def sample_code_alpaca(n: int = 5000) -> list[dict]:
    logger.info("Loading CodeAlpaca...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    prompt_key = "instruction" if "instruction" in ds.column_names else "prompt"
    return [{"prompt": ds[i][prompt_key], "source": "code_alpaca"} for i in indices]


def tokenize_prompt(tokenizer, prompt: str, enable_thinking: bool) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare Draft-OPD prompt pool")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Target model for tokenization")
    parser.add_argument("--output", default="opd_prompts.jsonl", help="Output jsonl path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gsm8k", type=int, default=2000)
    parser.add_argument("--n-math", type=int, default=5000)
    parser.add_argument("--n-aops", type=int, default=4000)
    parser.add_argument("--n-code", type=int, default=5000)
    parser.add_argument(
        "--enable-thinking", action="store_true", default=True,
        help="Enable thinking mode in chat template (default: True)",
    )
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    random.seed(args.seed)

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    all_prompts = []
    all_prompts.extend(sample_gsm8k(args.n_gsm8k))
    all_prompts.extend(sample_math(args.n_math))
    all_prompts.extend(sample_aops(args.n_aops))
    all_prompts.extend(sample_code_alpaca(args.n_code))

    logger.info("Total prompts collected: %d", len(all_prompts))
    random.shuffle(all_prompts)

    logger.info("Tokenizing with thinking=%s...", args.enable_thinking)
    written = 0
    with open(args.output, "w") as f:
        for item in all_prompts:
            try:
                input_ids = tokenize_prompt(
                    tokenizer, item["prompt"], args.enable_thinking
                )
                f.write(json.dumps({
                    "input_ids": input_ids,
                    "source": item["source"],
                }) + "\n")
                written += 1
            except Exception as e:
                logger.warning("Failed to tokenize prompt: %s", e)

    logger.info("Wrote %d prompts to %s", written, args.output)
    sources = {}
    for item in all_prompts:
        sources[item["source"]] = sources.get(item["source"], 0) + 1
    for source, count in sorted(sources.items()):
        logger.info("  %s: %d", source, count)


if __name__ == "__main__":
    main()
