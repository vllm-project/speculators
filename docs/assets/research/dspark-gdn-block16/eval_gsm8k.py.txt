"""Correctness and staggered diagnostic workloads, using vLLM's GSM8K helpers."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from tests.evals.gsm8k.gsm8k_eval import (
    _build_gsm8k_prompts,
    _score_gsm8k,
    get_answer_value,
)


async def evaluate(
    port: int, count: int, concurrency: int, stagger: bool
) -> tuple[list, list, list]:
    prompts, labels = _build_gsm8k_prompts(count, 5)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:

        async def request(i: int, prompt: str) -> dict:
            if stagger:
                await asyncio.sleep(i * 0.05)
            async with semaphore:
                body = {
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                async with session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions", json=body
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return {
                        "question": i,
                        "text": data["choices"][0]["message"]["content"],
                        "tokens": data["usage"]["completion_tokens"],
                        "finish_reason": data["choices"][0]["finish_reason"],
                    }

        records = await asyncio.gather(
            *(request(i, prompt) for i, prompt in enumerate(prompts))
        )
    return records, labels, prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--stagger", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    records, labels, _ = asyncio.run(
        evaluate(args.port, args.count, args.concurrency, args.stagger)
    )
    for record, label in zip(records, labels):
        record["label"] = label
        record["answer"] = get_answer_value(record["text"])
    score = _score_gsm8k(
        [r["text"] for r in records],
        [r["tokens"] for r in records],
        labels,
        5,
        1024,
        time.monotonic() - started,
    )
    args.output.write_text(json.dumps({"score": score, "records": records}, indent=2))
    print(json.dumps(score), flush=True)
    assert all(r["text"] and r["tokens"] > 0 for r in records), "Empty response"


if __name__ == "__main__":
    main()
