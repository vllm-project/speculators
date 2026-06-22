#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

DATASET_CONFIGS = {
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
        "default_splits": ["train"],
    },
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt", # Ultrachat actually uses 'messages'
        "default_splits": ["train_sft", "test_sft"],
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "prompt_field": "question",
        "default_splits": ["train"],
        "subset": "main",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate responses via vLLM Chat API.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default="ultrachat", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--splits", default=None, help="Comma-separated splits")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N new rows queued per split")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--language-filter", default=None)
    
    # New Arguments for RFC #584
    parser.add_argument("--output-mode", choices=["single", "bundle"], default="single")
    parser.add_argument("--outfile", default=None, help="Output JSONL path for 'single' mode")
    parser.add_argument("--output-dir", default="./output", help="Output directory for 'bundle' mode")
    
    parser.add_argument("--turn-mode", choices=["single", "multi"], default="single")
    parser.add_argument("--keep-length-finished", action="store_true", help="Keep length-finished responses")
    
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output")
    return parser.parse_args()

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    return name.replace(" ", "_").strip("._")

def get_primary_id(row: dict, idx: int) -> Tuple[str, str]:
    if "id" in row and row["id"]:
        return str(row["id"]), "id"
    if "uuid" in row and row["uuid"]:
        return str(row["uuid"]), "uuid"
    
    # Stable content hash fallback
    content = json.dumps(row, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest(), "hash"

def load_seen(paths: List[str]) -> Set[str]:
    seen = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    # New primary_id
                    if "metadata" in obj and "primary_id" in obj["metadata"]:
                        seen.add(str(obj["metadata"]["primary_id"]))
                    # Legacy fallback
                    key = obj.get("uuid") or obj.get("idx") or obj.get("id")
                    if key is not None:
                        seen.add(str(key))
                except json.JSONDecodeError:
                    pass
    return seen

async def detect_model(endpoint: str, timeout: float) -> str:
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")
    t = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=t) as session:
        async with session.get(models_endpoint) as response:
            data = await response.json()
            models = data.get("data", [])
            if models:
                return models[0]["id"]
            raise ValueError("No models found at endpoint")

async def generate_turn(session, endpoint, model, messages, max_tokens, args):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    
    retries = 0
    while retries <= args.max_retries:
        start_time = time.time()
        try:
            async with session.post(endpoint, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                return {
                    "content": message["content"],
                    "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
                    "finish_reason": choice.get("finish_reason"),
                    "latency": time.time() - start_time,
                    "usage": data.get("usage"),
                    "success": True
                }
        except Exception as e:
            retries += 1
            if retries > args.max_retries:
                return {"success": False, "error": repr(e)}
            await asyncio.sleep(args.retry_backoff ** retries)

def extract_user_turns(row, prompt_field):
    if "messages" in row and isinstance(row["messages"], list):
        return [m["content"] for m in row["messages"] if m["role"] == "user"]
    if "conversations" in row and isinstance(row["conversations"], list):
        return [m["value"] for m in row["conversations"] if m["from"] in ["human", "user"]]
    
    prompt = row.get(prompt_field)
    if prompt:
        return [prompt]
    return []

async def worker(sem, session, queue, args, out_fhs, endpoint, progress_bars, stats):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        split = item["split"]
        idx = item["idx"]
        primary_id = item["primary_id"]
        primary_id_source = item["primary_id_source"]
        user_turns = item["user_turns"]
        
        conversations = []
        metadata = {
            "idx": idx,
            "primary_id": primary_id,
            "primary_id_source": primary_id_source,
            "endpoint": endpoint,
            "split": split
        }
        
        status = "new_skipped"
        total_latency = 0
        
        for i, user_turn in enumerate(user_turns):
            conversations.append({"role": "user", "content": user_turn})
            
            res = await generate_turn(session, endpoint, args.model, conversations, args.max_tokens, args)
            if not res["success"]:
                metadata["error"] = res["error"]
                status = "error"
                break
            
            total_latency += res["latency"]
            conversations.append({"role": "assistant", "content": res["content"]})
            
            if res["finish_reason"] == "length":
                if not args.keep_length_finished:
                    status = "dropped_length"
                    break
                else:
                    # Keep but stop regenerating further turns
                    metadata["finish_reason"] = "length"
                    metadata["latency_s"] = round(total_latency, 3)
                    metadata["usage"] = res["usage"]
                    if res["reasoning_content"]:
                        metadata["reasoning_content"] = res["reasoning_content"]
                    status = "new_written_partial"
                    break
            
            # Successful turn
            metadata["finish_reason"] = res["finish_reason"]
            metadata["latency_s"] = round(total_latency, 3)
            metadata["usage"] = res["usage"]
            if res["reasoning_content"]:
                metadata["reasoning_content"] = res["reasoning_content"]
            
            if args.turn_mode == "single":
                status = "new_written"
                break
        else:
            if status != "error":
                status = "new_written"
                
        # Remap conversation format for output
        out_convs = []
        for msg in conversations:
            out_convs.append({
                "from": "human" if msg["role"] == "user" else "gpt",
                "value": msg["content"]
            })
            
        output = {
            "id": primary_id,
            "conversations": out_convs,
            "metadata": metadata
        }
        
        if status in ["new_written", "new_written_partial"]:
            out_fh = out_fhs[split] if args.output_mode == "bundle" else out_fhs["single"]
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats[split]["written"] += 1
            stats["total"]["written"] += 1
        elif status == "error":
            stats[split]["error"] += 1
            stats["total"]["error"] += 1
        else:
            stats[split]["dropped"] += 1
            stats["total"]["dropped"] += 1
            
        pb = progress_bars.get(split) or progress_bars.get("single")
        if pb:
            pb.update(1)
            
        queue.task_done()

async def main():
    args = parse_args()
    if not args.model:
        args.model = await detect_model(args.endpoint, args.request_timeout)

    ds_config = DATASET_CONFIGS[args.dataset]
    splits = args.splits.split(",") if args.splits else ds_config["default_splits"]
    
    out_fhs = {}
    output_paths = []
    
    if args.output_mode == "single":
        if not args.outfile:
            mname = sanitize_filename(args.model.split("/")[-1])
            args.outfile = f"{args.dataset}_{mname}.jsonl"
        out_fhs["single"] = open(args.outfile, "a", encoding="utf-8")
        output_paths.append(args.outfile)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        mname = sanitize_filename(args.model.split("/")[-1])
        for split in splits:
            path = os.path.join(args.output_dir, f"{args.dataset}_{split}_{mname}.jsonl")
            out_fhs[split] = open(path, "a", encoding="utf-8")
            output_paths.append(path)
            
    seen_ids = load_seen(output_paths) if args.resume else set()
    
    queue = asyncio.Queue(maxsize=args.concurrency * 4)
    semaphore = asyncio.Semaphore(args.concurrency)
    
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=None)
    
    stats = defaultdict(lambda: {"written": 0, "dropped": 0, "pre_existing": 0, "error": 0})
    progress_bars = {}
    
    if args.output_mode == "single":
        progress_bars["single"] = tqdm(desc="Total Processing", unit="req", dynamic_ncols=True)
    else:
        for split in splits:
            progress_bars[split] = tqdm(desc=f"Split: {split}", unit="req", dynamic_ncols=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        workers = [asyncio.create_task(worker(semaphore, session, queue, args, out_fhs, args.endpoint, progress_bars, stats)) for _ in range(args.concurrency)]
        
        for split in splits:
            dataset = load_dataset(ds_config["id"], name=ds_config.get("subset"), split=split, streaming=True)
            new_queued = 0
            
            for idx, row in enumerate(dataset):
                if args.limit and new_queued >= args.limit:
                    break
                    
                if args.language_filter and row.get("language") != args.language_filter:
                    continue
                    
                user_turns = extract_user_turns(row, ds_config["prompt_field"])
                if not user_turns:
                    continue
                    
                pid, psrc = get_primary_id(row, idx)
                if pid in seen_ids:
                    stats[split]["pre_existing"] += 1
                    stats["total"]["pre_existing"] += 1
                    continue
                    
                await queue.put({
                    "split": split,
                    "idx": idx,
                    "primary_id": pid,
                    "primary_id_source": psrc,
                    "user_turns": user_turns,
                })
                new_queued += 1
                
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        
    for fh in out_fhs.values():
        fh.close()
        
    for pb in progress_bars.values():
        pb.close()
        
    print("\nRun Summary:")
    for split in splits:
        s = stats[split]
        processed = s["pre_existing"] + s["written"] + s["dropped"] + s["error"]
        print(f"[{split}] Processed: {processed} (Pre-existing: {s['pre_existing']}, Written: {s['written']}, Dropped: {s['dropped']}, Errors: {s['error']})")

if __name__ == "__main__":
    asyncio.run(main())
