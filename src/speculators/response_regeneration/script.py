#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

import aiohttp
from datasets import load_dataset

DATASET_CONFIGS = {
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
    },
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt",
    },
}


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate responses from Magpie instructions via vLLM Chat API."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM OpenAI-compatible Chat Completions endpoint (used if --ports is not set)",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1",
        help="Base host for vLLM servers (used with --ports, e.g. http://127.0.0.1)",
    )
    parser.add_argument(
        "--ports",
        default=None,
        help=(
            "Comma-separated list of ports for multiple vLLM servers "
            "(e.g. '8000,8001,8002'). If set, overrides --endpoint and "
            "builds endpoints as {host}:{port}/v1/chat/completions."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        default="ultrachat",
        choices=["magpie", "ultrachat"],
        help="Dataset to process (magpie or ultrachat)",
    )
    parser.add_argument("--split", default="train_sft", help="Dataset split")
    parser.add_argument(
        "--subset",
        default=None,
        help="(unused) kept for symmetry with other scripts",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent requests (total across all servers)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="max_tokens for generation",
    )
    parser.add_argument(
        "--outfile",
        default="ultrachat_qwen3_vl.jsonl",
        help="Where to write JSONL results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already in outfile (by uuid or idx)",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Only process rows where language==this (e.g., EN)",
    )
    return parser.parse_args()


def load_seen(path: str):
    """Load previously processed record IDs from output file for resume functionality."""
    seen = set()
    if not os.path.isfile(path):
        return seen

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                key = obj.get("uuid") or obj.get("idx")
                if key is not None:
                    seen.add(str(key))
            except Exception:
                continue
    return seen


async def detect_model(endpoint: str) -> str:
    """Automatically detect the model name from the vLLM server."""
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")

    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(models_endpoint) as response:
                data = await response.json()
                models = data.get("data", [])
                if models:
                    model_name = models[0]["id"]
                    print(f"Auto-detected model: {model_name}")
                    return model_name
                else:
                    raise ValueError("No models found at endpoint")
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}\n"
            f"Please specify model with --model argument"
        )


async def worker(
    name: int,
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[Dict[str, Any]]",
    args,
    out_fh,
    endpoints: List[str],
):
    """Worker that pulls items from queue and sends them to one of multiple endpoints."""
    endpoint = endpoints[name % len(endpoints)]

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": args.max_tokens,
        }

        start_time = time.time()
        try:
            async with sem:
                async with session.post(endpoint, json=payload) as response:
                    data = await response.json()

            choice = data["choices"][0]
            generated_text = choice["message"]["content"]
            reasoning_content = choice["message"].get("reasoning_content")
            finish_reason = choice.get("finish_reason")
            latency = time.time() - start_time

            # Format output in conversations structure
            metadata = {
                "idx": idx,
                "finish_reason": finish_reason,
                "latency_s": round(latency, 3),
                "usage": data.get("usage"),
                "endpoint": endpoint,
            }

            # Only include reasoning_content if it exists
            if reasoning_content is not None:
                metadata["reasoning_content"] = reasoning_content

            output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [
                    {
                        "from": "human",
                        "value": item["prompt"]
                    },
                    {
                        "from": "gpt",
                        "value": generated_text
                    }
                ],
                "metadata": metadata
            }
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
        except Exception as e:
            error_output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [
                    {
                        "from": "human",
                        "value": item["prompt"]
                    }
                ],
                "metadata": {
                    "idx": idx,
                    "error": repr(e),
                    "endpoint": endpoint,
                }
            }
            out_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            out_fh.flush()
        finally:
            queue.task_done()


async def main():
    """Main async function to process dataset through vLLM endpoints."""
    args = parse_args()

    # Build list of endpoints (one per vLLM server)
    if args.ports:
        ports = [port.strip() for port in args.ports.split(",") if port.strip()]
        if not ports:
            raise ValueError("No valid ports parsed from --ports argument.")
        endpoints = [f"{args.host}:{port}/v1/chat/completions" for port in ports]
    else:
        endpoints = [args.endpoint]

    print("Using endpoints:")
    for endpoint in endpoints:
        print(f"  {endpoint}")

    # Auto-detect model if not specified
    if args.model is None:
        args.model = await detect_model(endpoints[0])

    print(f"Using model: {args.model}")

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset_id = dataset_config["id"]
    prompt_field = dataset_config["prompt_field"]
    print(f"Using dataset: {dataset_id}")
    print(f"Prompt field: {prompt_field}")
    print()

    seen_ids = load_seen(args.outfile) if args.resume else set()
    dataset = load_dataset(dataset_id, split=args.split, streaming=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    semaphore = asyncio.Semaphore(args.concurrency)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    connector = aiohttp.TCPConnector(
        limit=None, force_close=False, enable_cleanup_closed=True
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        with open(args.outfile, "a", encoding="utf-8") as output_file:
            workers = [
                asyncio.create_task(
                    worker(i, semaphore, session, queue, args, output_file, endpoints)
                )
                for i in range(args.concurrency)
            ]

            processed_count = 0
            for index, row in enumerate(dataset):
                if args.limit is not None and processed_count >= args.limit:
                    break

                if args.language_filter and row.get("language") != args.language_filter:
                    continue

                prompt = row.get(prompt_field)
                if not prompt:
                    continue

                uuid = row.get("uuid")
                key = str(uuid or index)
                if key in seen_ids:
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "uuid": uuid,
                        "prompt": prompt,
                    }
                )
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

