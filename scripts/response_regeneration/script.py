#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

DATASET_CONFIGS = {
    # 300K filtered synthetic instructions generated via Magpie from Llama-3.1.
    # Single "train" split with an "instruction" field.
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
        "default_split": "train",
        "id_field": "uuid",
    },
    # 200K multi-turn dialogues covering a broad range of topics. The "train_sft"
    # split contains the SFT-ready subset with a "prompt" field.
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt",
        "default_split": "train_sft",
        "id_field": "prompt_id",
    },
    # Grade-school math word problems with step-by-step solutions. Uses the "main"
    # subset. Good for evaluating mathematical reasoning capabilities.
    "gsm8k": {
        "id": "openai/gsm8k",
        "prompt_field": "question",
        "default_split": "train",
        "subset": "main",
    },
    # 20K code generation prompts based on the Stanford Alpaca format. Each row has a
    # plain string "prompt" field describing a coding task. Compact and code-focused.
    "code_alpaca": {
        "id": "HuggingFaceH4/CodeAlpaca_20K",
        "prompt_field": "prompt",
        "default_split": "train",
    },
    # NVIDIA's large-scale post-training dataset covering multiple domains. Available
    # splits: chat, math, code, stem. Uses a messages format with role/content pairs.
    # Select domain via --split (defaults to "chat").
    "nemotron": {
        "id": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "prompt_field": "messages",
        "default_split": "chat",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # Allen AI's SFT mixture used to train Tulu 3. Contains ~939K examples spanning
    # diverse tasks and sources. Uses a messages format with role/content pairs.
    "tulu3": {
        "id": "allenai/tulu-3-sft-mixture",
        "prompt_field": "messages",
        "default_split": "train",
        "id_field": "id",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # ~529K real user conversations collected from ChatGPT and GPT-4. Useful for
    # capturing natural user interaction patterns. Uses a "conversation" field with
    # role/content pairs.
    "wildchat": {
        "id": "allenai/WildChat",
        "prompt_field": "conversation",
        "default_split": "train",
        "id_field": "conversation_id",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's Cascade 2 SFT data spanning 8 domains. Each domain is a separate
    # subset: chat, conversational_agent, instruction_following, math, safety,
    # science, swe, terminal_agent. Select domain via --subset (defaults to "chat").
    "nemotron_cascade": {
        "id": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "prompt_field": "messages",
        "default_split": "train",
        "subset": "chat",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's instruction-following and chat SFT dataset with synthetic dialogues
    # from multiple frontier models. Available in two splits: "reasoning_off"
    # (default) and "reasoning_on" for chain-of-thought style responses.
    "nemotron_ifchat": {
        "id": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
        "prompt_field": "messages",
        "default_split": "reasoning_off",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's instruction-following chat v1 dataset with structured outputs.
    "nemotron_ifchat_v1": {
        "id": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "prompt_field": "messages",
        "default_split": "structured_outputs",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's agentic SFT dataset covering interactive agents, tool calling,
    # and search. Select domain via --split.
    "nemotron_agentic": {
        "id": "nvidia/Nemotron-SFT-Agentic-v2",
        "prompt_field": "messages",
        "default_split": "interactive_agent",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's competitive programming dataset v2 with Python and C++ solutions.
    "nemotron_competitive_v2": {
        "id": "nvidia/Nemotron-SFT-Competitive-Programming-v2",
        "prompt_field": "messages",
        "default_split": "competitive_coding_python",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's competitive programming dataset v1 with Infinibyte-generated data.
    "nemotron_competitive_v1": {
        "id": "nvidia/Nemotron-Competitive-Programming-v1",
        "prompt_field": "messages",
        "default_split": "infinibyte_part00",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's math SFT dataset v2 with high/medium/low difficulty splits.
    "nemotron_math": {
        "id": "nvidia/Nemotron-Math-v2",
        "prompt_field": "messages",
        "default_split": "high_part00",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's science dataset with multiple-choice and reasoning Q&A splits.
    "nemotron_science": {
        "id": "nvidia/Nemotron-Science-v1",
        "prompt_field": "messages",
        "default_split": "MCQ",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # NVIDIA's software engineering SFT dataset with agentless trajectories.
    "nemotron_swe": {
        "id": "nvidia/Nemotron-SFT-SWE-v2",
        "prompt_field": "messages",
        "default_split": "agentless",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # ~10K long-context instruction-following samples (8k-64k tokens). Useful for
    # training speculators on long-form generation patterns.
    "longalign": {
        "id": "zai-org/LongAlign-10k",
        "prompt_field": "messages",
        "default_split": "train",
        "id_field": "id",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # ~1.4M instruction samples blending 8 source datasets (math, code, chat,
    # instruction-following). Uses ShareGPT-style conversations with from/value
    # fields and "human"/"gpt" roles.
    "open_perfectblend": {
        "id": "mlabonne/open-perfectblend",
        "prompt_field": "conversations",
        "default_split": "train",
        "messages_role_field": "from",
        "messages_content_field": "value",
        "messages_user_value": "human",
    },
    # --- Individual sources from open_perfectblend ---
    # 395K augmented math questions with detailed solutions.
    "metamathqa": {
        "id": "meta-math/MetaMathQA",
        "prompt_field": "query",
        "default_split": "train",
    },
    # 289K instruction-response pairs spanning math, code, and logic tasks.
    "ultrainteract": {
        "id": "openbmb/UltraInteract_sft",
        "prompt_field": "instruction",
        "default_split": "train",
        "id_field": "id",
    },
    # 200K math word problems with step-by-step solutions.
    "orca_math": {
        "id": "microsoft/orca-math-word-problems-200k",
        "prompt_field": "question",
        "default_split": "train",
    },
    # 187K preference-ranked prompts. Uses the SFT split with plain-string prompts.
    "ultrafeedback": {
        "id": "HuggingFaceH4/ultrafeedback_binarized",
        "prompt_field": "prompt",
        "default_split": "train_sft",
        "id_field": "prompt_id",
    },
    # 111K evolved code instruction-output pairs.
    "evol_codealpaca": {
        "id": "theblackcat102/evol-codealpaca-v1",
        "prompt_field": "instruction",
        "default_split": "train",
    },
    # 61K auto-generated instruction-following examples.
    "autoif": {
        "id": "Post-training-Data-Flywheel/AutoIF-instruct-61k",
        "prompt_field": "messages",
        "default_split": "train",
        "id_field": "conversation_id",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # 57K human-preference conversations from LMSYS Arena in ShareGPT format.
    "lmsys_arena": {
        "id": "mlabonne/lmsys-arena-human-preference-55k-sharegpt",
        "prompt_field": "conversations",
        "default_split": "train",
        "messages_role_field": "from",
        "messages_content_field": "value",
        "messages_user_value": "human",
    },
    # 1M instruction/chat examples covering diverse tasks. ShareGPT-style
    # conversations with from/value fields.
    "openhermes": {
        "id": "teknium/OpenHermes-2.5",
        "prompt_field": "conversations",
        "default_split": "train",
        "messages_role_field": "from",
        "messages_content_field": "value",
        "messages_user_value": "human",
    },
    # High-quality verified R1-style math reasoning traces. Available subsets:
    # default (94K), extended (131K), all (225K). Select via --subset.
    "openr1_math": {
        "id": "open-r1/OpenR1-Math-220k",
        "prompt_field": "messages",
        "default_split": "train",
        "id_field": "uuid",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # 72K math problems with tool-integrated reasoning solutions.
    "numinamath": {
        "id": "AI-MO/NuminaMath-TIR",
        "prompt_field": "messages",
        "default_split": "train",
        "messages_role_field": "role",
        "messages_content_field": "content",
    },
    # Competitive coding problems with R1-style reasoning traces. Multiple
    # subsets available; defaults to "solutions". Select via --subset.
    "codeforces_cots": {
        "id": "open-r1/codeforces-cots",
        "prompt_field": "prompt",
        "default_split": "train",
        "subset": "solutions",
        "id_field": "id",
    },
    # 10K+ Codeforces problems through 2025. Problem descriptions used as
    # prompts. Select subset via --subset (default, verifiable, verifiable-prompts).
    "codeforces": {
        "id": "open-r1/codeforces",
        "prompt_field": "description",
        "default_split": "train",
        "id_field": "id",
    },
    # 25K competitive programming tasks with test cases.
    "taco": {
        "id": "BAAI/TACO",
        "prompt_field": "question",
        "default_split": "train",
    },
    # 60K verified function-calling examples with query/tools/answers format.
    "xlam_function_calling": {
        "id": "Salesforce/xlam-function-calling-60k",
        "prompt_field": "query",
        "default_split": "train",
    },
    # 5K multi-turn tool/agent trajectories. ShareGPT-style conversations.
    "apigen_mt": {
        "id": "Salesforce/APIGen-MT-5k",
        "prompt_field": "conversations",
        "default_split": "train",
        "messages_role_field": "from",
        "messages_content_field": "value",
        "messages_user_value": "human",
    },
    # 67K realistic SWE agent traces from OpenHands on SWE-bench tasks.
    "swe_rebench": {
        "id": "nebius/SWE-rebench-openhands-trajectories",
        "prompt_field": "trajectory",
        "default_split": "train",
        "id_field": "trajectory_id",
        "messages_role_field": "role",
        "messages_content_field": "content",
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
        help="vLLM OpenAI-compatible Chat Completions endpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        default="ultrachat",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (defaults to dataset-specific split)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Dataset subset/config name "
            "(auto-detected from dataset config if not specified)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="max_tokens for generation",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output JSONL path (auto-generated if not specified)",
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
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Buffer size for streaming shuffle (default: 10000)",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def load_seen(path: str):
    """Load previously processed record IDs from output file."""
    seen = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("id") or obj.get("uuid") or obj.get("idx")
            if key is not None:
                seen.add(str(key))
    return seen


async def detect_model(endpoint: str) -> str:
    """Automatically detect the model name from the vLLM server."""
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")

    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(models_endpoint) as response,
        ):
            data = await response.json()
            models = data.get("data", [])
            if models:
                model_name = models[0]["id"]
                print(f"Auto-detected model: {model_name}")
                return model_name
            raise ValueError("No models found at endpoint")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}\n"
            f"Please specify model with --model argument"
        ) from e


async def worker(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    args,
    out_fh,
    endpoint: str,
    pbar: tqdm = None,
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
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
            async with sem, session.post(endpoint, json=payload) as response:
                data = await response.json()

            choice = data["choices"][0]
            message = choice["message"]
            generated_text = message["content"]
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is None:
                reasoning_content = message.get("reasoning")
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
                    {"from": "human", "value": item["prompt"]},
                    {"from": "gpt", "value": generated_text},
                ],
                "metadata": metadata,
            }
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
        except Exception as e:  # noqa: BLE001
            error_output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [{"from": "human", "value": item["prompt"]}],
                "metadata": {
                    "idx": idx,
                    "error": repr(e),
                    "endpoint": endpoint,
                },
            }
            out_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            out_fh.flush()
        finally:
            if pbar is not None:
                pbar.update(1)
            queue.task_done()


async def main():  # noqa: C901, PLR0915, PLR0912
    """Main async function to process dataset through vLLM endpoints."""
    args = parse_args()

    endpoint = args.endpoint
    print(f"Using endpoint: {endpoint}")

    # Auto-detect model if not specified
    if args.model is None:
        args.model = await detect_model(endpoint)

    print(f"Using model: {args.model}")

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset_id = dataset_config["id"]
    prompt_field = dataset_config["prompt_field"]

    # Use dataset-specific defaults if not provided
    split = args.split if args.split is not None else dataset_config["default_split"]
    subset = args.subset if args.subset is not None else dataset_config.get("subset")

    # Generate output filename if not specified
    if args.outfile is None:
        # Extract simple model name from full path
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        model_name = sanitize_filename(model_name)
        parts = [args.dataset]
        if subset:
            parts.append(sanitize_filename(subset))
        parts.append(sanitize_filename(split))
        parts.append(model_name)
        args.outfile = "_".join(parts) + ".jsonl"

    print(f"Using dataset: {dataset_id}")
    print(f"Subset: {subset}")
    print(f"Split: {split}")
    print(f"Prompt field: {prompt_field}")
    print(f"Output file: {args.outfile}")
    print()

    seen_ids = load_seen(args.outfile) if args.resume else set()
    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)

    if args.shuffle:
        dataset = dataset.shuffle(
            seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size
        )
        print(
            f"Shuffling with seed={args.shuffle_seed},"
            f" buffer={args.shuffle_buffer_size}"
        )

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
        with open(args.outfile, "a", encoding="utf-8") as output_file:  # noqa: ASYNC230
            pbar = tqdm(
                total=args.limit,
                desc="Generating responses",
                unit=" samples",
                dynamic_ncols=True,
            )

            workers = [
                asyncio.create_task(
                    worker(semaphore, session, queue, args, output_file, endpoint, pbar)
                )
                for _ in range(args.concurrency)
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

                if isinstance(prompt, list):
                    role_field = dataset_config.get("messages_role_field", "role")
                    content_field = dataset_config.get(
                        "messages_content_field", "content"
                    )
                    user_value = dataset_config.get("messages_user_value", "user")
                    user_msgs = [
                        m[content_field]
                        for m in prompt
                        if m.get(role_field) == user_value
                    ]
                    if not user_msgs:
                        continue
                    prompt = user_msgs[0]

                id_field = dataset_config.get("id_field")
                row_id = row.get(id_field) if id_field else None
                key = str(row_id if row_id is not None else index)
                if key in seen_ids:
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "uuid": row_id,
                        "prompt": prompt,
                    }
                )
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)
            pbar.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
