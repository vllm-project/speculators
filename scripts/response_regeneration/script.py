#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
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
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
        "default_split": "train",
    },
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt",
        "default_split": "train_sft",
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "prompt_field": "question",
        "default_split": "train",
        "subset": "main",
    },
}

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0


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
        help="Skip rows already in outfile (by stable primary id)",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Only process rows where language==this (e.g., EN)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Max retry attempts per request on transient failure "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help=(
            "Initial retry backoff in seconds; doubles after each failed attempt "
            f"(default: {DEFAULT_RETRY_BACKOFF})"
        ),
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def extract_turns(row, prompt_field):
    """Extract ordered system/user turns from a dataset row.

    Multi-turn conversations are read from a ``messages`` or ``conversations``
    field (either the role/content or from/value schema), preserving any system
    prompt and dropping the original assistant turns so they can be regenerated.
    Rows without a usable conversation fall back to a single user turn taken
    from ``prompt_field``.
    """
    convs = row.get("messages")
    if not (isinstance(convs, list) and convs):
        convs = row.get("conversations")

    if isinstance(convs, list) and convs:
        turns = []
        for m in convs:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            if not content:
                continue
            if role == "system":
                turns.append({"role": "system", "content": content})
            elif role in ("user", "human"):
                turns.append({"role": "user", "content": content})
            # original assistant/gpt turns are dropped and regenerated
        if any(turn["role"] == "user" for turn in turns):
            return turns
        # no usable user turn: fall through to the prompt_field fallback

    prompt = row.get(prompt_field)
    if prompt:
        return [{"role": "user", "content": prompt}]
    return []


def _is_present(value: Any) -> bool:
    """Return True for a usable identifier (not None / not empty string)."""
    return value is not None and str(value) != ""


def _content_hash(row: dict[str, Any]) -> str:
    """Deterministic hash of a row, used as a last-resort stable identifier."""
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return "hash_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_identifier(row: dict[str, Any]) -> tuple[str, str]:
    """Return a stable ``(id, source)`` for a dataset row.

    Prefers an explicit ``id``/``uuid``, then a nested ``metadata`` id, and
    finally a content hash. Unlike a streaming enumeration index, this key does
    not shift when ``--limit``/``--language-filter`` or the input order change,
    so ``--resume`` stays correct across runs.
    """
    for field in ("id", "uuid"):
        value = row.get(field)
        if _is_present(value):
            return str(value), field

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for field in ("id", "uuid", "sample_id", "idx", "index"):
            value = metadata.get(field)
            if _is_present(value):
                return str(value), f"metadata.{field}"

    return _content_hash(row), "content_hash"


def _output_seen_ids(obj: dict[str, Any]) -> set[str]:
    """Extract resume keys from one previously written output row.

    Prefers ``metadata.primary_id`` (written by this script); falls back to
    legacy top-level ``id``/``uuid`` and nested metadata ids so output files
    written before this change still resume for rows that carried a real id.
    """
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        primary_id = metadata.get("primary_id")
        if _is_present(primary_id):
            return {str(primary_id)}

    ids: set[str] = set()
    for field in ("id", "uuid"):
        value = obj.get(field)
        if _is_present(value):
            ids.add(str(value))
    if isinstance(metadata, dict):
        for field in ("id", "uuid", "sample_id", "idx", "index"):
            value = metadata.get(field)
            if _is_present(value):
                ids.add(str(value))
    return ids


def load_seen(path: str) -> set[str]:
    """Load previously processed primary IDs from output file."""
    seen: set[str] = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.update(_output_seen_ids(obj))
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


async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> dict[str, Any]:
    """POST one chat-completion request and return the parsed response.

    A non-2xx reply has no ``choices`` array, so raise with the status and a
    short body; otherwise the caller records a bare ``KeyError('choices')`` and
    the real cause is lost. Transient failures (network errors or non-2xx
    replies) are retried up to ``max_retries`` times with exponential backoff;
    the semaphore is released between attempts so a backing-off request does not
    hold a concurrency slot.
    """
    total_attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            async with session.post(endpoint, json=payload) as response:
                if not response.ok:
                    body = (await response.text())[:500]
                    raise RuntimeError(
                        f"HTTP {response.status} from {endpoint}: {body}"
                    )
                return await response.json()
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt >= total_attempts:
                break
            await asyncio.sleep(retry_backoff * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"chat request failed after {total_attempts} attempt(s)"
    ) from last_error


async def worker(
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    args,
    out_fh,
    err_fh,
    endpoint: str,
    progress,
    stats: dict[str, int],
):
    """Worker that pulls conversations from the queue and regenerates them.

    Each user turn is sent to the endpoint with the freshly generated prefix
    (system + regenerated history); the original assistant turns are discarded
    and replaced by the model's responses. Single-turn rows are the degenerate
    case of one user turn.
    """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        turns = item["turns"]

        # The API prefix (role/content) and the output conversation (from/value)
        # are built in lockstep as we walk the turns.
        prefix: list[dict[str, Any]] = []
        out_convs: list[dict[str, Any]] = []
        finish_reasons: list[str | None] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        start_time = time.time()
        try:
            for turn in turns:
                if turn["role"] == "system":
                    prefix.append({"role": "system", "content": turn["content"]})
                    out_convs.append({"from": "system", "value": turn["content"]})
                    continue

                prefix.append({"role": "user", "content": turn["content"]})
                out_convs.append({"from": "human", "value": turn["content"]})

                payload = {
                    "model": args.model,
                    "messages": prefix,
                    "max_tokens": args.max_tokens,
                }
                data = await _post_chat(
                    session,
                    endpoint,
                    payload,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )

                choice = data["choices"][0]
                message = choice["message"]
                generated_text = message.get("content")
                reasoning_content = message.get("reasoning_content")
                if reasoning_content is None:
                    reasoning_content = message.get("reasoning")
                finish_reasons.append(choice.get("finish_reason"))
                turn_usage = data.get("usage") or {}
                for key in usage:
                    usage[key] += turn_usage.get(key) or 0

                # Empty content (e.g. truncated mid-reasoning) would corrupt the
                # next turn's prefix and emit a null target; fail the conversation.
                if not generated_text:
                    raise ValueError(f"empty assistant content (turn {len(out_convs)})")

                prefix.append({"role": "assistant", "content": generated_text})
                gpt_turn = {"from": "gpt", "value": generated_text}
                # Stored per turn: data prep reads it here, not top-level metadata.
                if reasoning_content is not None:
                    gpt_turn["reasoning_content"] = reasoning_content
                out_convs.append(gpt_turn)

            metadata = {
                "primary_id": item["primary_id"],
                "primary_id_source": item["primary_id_source"],
                "idx": idx,
                "finish_reasons": finish_reasons,
                "latency_s": round(time.time() - start_time, 3),
                "usage": usage,
                "endpoint": endpoint,
            }

            output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": out_convs,
                "metadata": metadata,
            }
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["ok"] += 1
        except Exception as e:  # noqa: BLE001
            # Failures go to a separate error file, not the training output; an
            # in-band marker would be invisible (the pipeline drops metadata).
            error_output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": out_convs,
                "metadata": {
                    "primary_id": item["primary_id"],
                    "primary_id_source": item["primary_id_source"],
                    "idx": idx,
                    "error": repr(e),
                    "endpoint": endpoint,
                },
            }
            err_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            err_fh.flush()
            stats["errors"] += 1
        finally:
            progress.set_postfix(
                ok=stats["ok"],
                errors=stats["errors"],
                refresh=False,
            )
            progress.update(1)
            queue.task_done()


async def main():
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
        args.outfile = f"{args.dataset}_{model_name}.jsonl"

    # Failed / partial conversations are written here instead of the training file.
    base, ext = os.path.splitext(args.outfile)
    error_outfile = f"{base}.errors{ext or '.jsonl'}"

    print(f"Using dataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Prompt field: {prompt_field}")
    print(f"Output file: {args.outfile}")
    print(f"Error file: {error_outfile}")
    print()

    seen_ids = load_seen(args.outfile) if args.resume else set()
    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)

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
        with (
            open(args.outfile, "a", encoding="utf-8") as output_file,  # noqa: ASYNC230
            open(error_outfile, "a", encoding="utf-8") as error_file,  # noqa: ASYNC230
            tqdm(
                total=args.limit,
                desc="Generating responses",
                unit="sample",
                dynamic_ncols=True,
            ) as progress,
        ):
            stats = {"ok": 0, "errors": 0}
            workers = [
                asyncio.create_task(
                    worker(
                        session,
                        queue,
                        args,
                        output_file,
                        error_file,
                        endpoint,
                        progress,
                        stats,
                    )
                )
                for _ in range(args.concurrency)
            ]

            processed_count = 0
            for index, row in enumerate(dataset):
                if args.limit is not None and processed_count >= args.limit:
                    break

                if args.language_filter and row.get("language") != args.language_filter:
                    continue

                turns = extract_turns(row, prompt_field)
                # extract_turns returns [] when there is no usable user turn.
                if not turns:
                    continue

                uuid = row.get("uuid")
                primary_id, primary_id_source = _primary_identifier(row)
                if primary_id in seen_ids:
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "uuid": uuid,
                        "primary_id": primary_id,
                        "primary_id_source": primary_id_source,
                        "turns": turns,
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
