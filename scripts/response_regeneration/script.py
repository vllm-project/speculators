#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from typing import Any

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

from speculators.data_generation.configs import DATASET_CONFIGS, DatasetConfig
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    InvalidResponseError,
    with_retries,
)

# Image parts: the Chat API rejects them, and output rows keep no pixel data.
MULTIMODAL_DATASETS = {"sharegpt4v_coco"}
REGEN_DATASETS = [name for name in DATASET_CONFIGS if name not in MULTIMODAL_DATASETS]


def _dataset_choice(name: str) -> str:
    """Reject multimodal presets with a reason, not a bare invalid choice."""
    if name in MULTIMODAL_DATASETS:
        raise argparse.ArgumentTypeError(
            f"{name!r} is multimodal; on-policy regeneration does not support "
            "images yet. Use it off-policy with `prepare-data`."
        )
    return name


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate dataset responses via a vLLM Chat API endpoint."
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
        type=_dataset_choice,
        choices=REGEN_DATASETS,
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
        "--sampling-params",
        default=None,
        help=(
            "JSON object merged into each chat-completion request, "
            'e.g. \'{"temperature": 0.6, "top_p": 0.95, "seed": 0}\''
        ),
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
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    try:
        args.sampling_params = (
            json.loads(args.sampling_params) if args.sampling_params else {}
        )
    except json.JSONDecodeError as e:
        parser.error(f"--sampling-params is not valid JSON: {e}")
    if not isinstance(args.sampling_params, dict):
        parser.error("--sampling-params must be a JSON object")
    return args


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def extract_turns(
    row: dict[str, Any], prompt_field: str | None
) -> list[dict[str, Any]]:
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


def prepare_row(row: dict[str, Any], config: DatasetConfig) -> list[dict[str, Any]]:
    """Extract regeneration turns from a raw dataset row, ``[]`` to skip it.

    Mirrors off-policy ingestion: ``filter_fn`` sees the raw row, and
    ``normalize_fn`` is merged over it (HF ``map`` semantics keep raw columns).
    """
    if config.filter_fn is not None and not config.filter_fn(row):
        return []
    if config.normalize_fn is not None:
        row = {**row, **config.normalize_fn(row)}
    return extract_turns(row, config.prompt_field)


def _is_present(value: Any) -> bool:
    """Return True for a usable identifier (not None / not empty string)."""
    return value not in (None, "")


def _content_hash(row: dict[str, Any]) -> str:
    """Deterministic hash of a row, used when it has no explicit id."""
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return "hash_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_identifier(row: dict[str, Any]) -> str:
    """Return a stable primary id for a dataset row.

    Prefers an explicit ``id``/``uuid``; otherwise a deterministic content hash.
    Unlike a streaming enumeration index, this key does not shift when
    ``--limit``/``--language-filter`` or the input order change, so ``--resume``
    stays correct across runs.
    """
    for field in ("id", "uuid"):
        value = row.get(field)
        if _is_present(value):
            return str(value)
    return _content_hash(row)


def load_seen(path: str) -> set[str]:
    """Load previously completed conversation ids from the output file.

    A conversation fans out to one row per assistant turn, whose ``id`` carries a
    ``_turn<N>`` suffix; the conversation's own :func:`_primary_identifier` is
    kept alongside it as ``primary_id``. Resume keys on that, since the suffixed
    ids never match a recomputed one. Rows are written only after every turn
    succeeds, so one row is enough to mark the conversation done.

    ``id`` is the fallback for output files written before the fan-out, where the
    top-level ``id`` *was* the primary identifier.
    """
    seen: set[str] = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("primary_id")
            if not _is_present(key):
                key = obj.get("id")
            if _is_present(key):
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


# Transient statuses worth retrying: request timeout, conflict, too-early, and
# rate limiting, plus all 5xx. Other non-2xx replies (e.g. 400/401/404) are
# permanent config/client errors and fail fast.
SERVER_ERROR_STATUS = 500
RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}


@with_retries
async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST one chat-completion request and return the parsed response.

    Wrapped by ``with_retries`` (adds a ``max_retries`` kwarg): transient
    failures — network errors and transient HTTP statuses (408/409/425/429/5xx)
    — are retried with exponential backoff. Permanent non-2xx replies (e.g.
    400/404) raise ``InvalidResponseError``, which ``with_retries`` never
    retries, so they fail fast. A non-2xx reply is surfaced with its status and
    a short body so the caller does not record a bare ``KeyError('choices')``.
    """
    async with session.post(endpoint, json=payload) as response:
        if not response.ok:
            body = (await response.text())[:500]
            message = f"HTTP {response.status} from {endpoint}: {body}"
            # Retry transient statuses (408/409/425/429/5xx); fail fast otherwise.
            if (
                response.status >= SERVER_ERROR_STATUS
                or response.status in RETRYABLE_HTTP_STATUSES
            ):
                raise RuntimeError(message)
            raise InvalidResponseError(message)
        return await response.json()


def build_boundary_sample(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Build one training sample: prompt (loss_mask 0) + generated tokens (1).

    The generation boundary is the mask -- no ``{% generation %}`` markers, no regex.
    """
    input_ids = [*prompt_token_ids, *completion_token_ids]
    loss_mask = [0] * len(prompt_token_ids) + [1] * len(completion_token_ids)
    return input_ids, loss_mask


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
    """Regenerate each queued conversation into pre-tokenized training samples.

    One sample per assistant turn: the prompt the target conditioned on
    (loss_mask 0) followed by the tokens it generated (1).
    """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        turns = item["turns"]
        conv_id = item["primary_id"]

        prefix: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        try:
            for turn in turns:
                if turn["role"] == "system":
                    prefix.append({"role": "system", "content": turn["content"]})
                    continue

                prefix.append({"role": "user", "content": turn["content"]})

                payload = {
                    **args.sampling_params,
                    "model": args.model,
                    "messages": prefix,
                    "max_tokens": args.max_tokens,
                    "return_token_ids": True,  # prompt_token_ids + completion token_ids
                }
                data = await _post_chat(
                    session,
                    endpoint,
                    payload,
                    max_retries=args.max_retries,
                )

                choice = data["choices"][0]
                generated_text = choice["message"].get("content")

                # Empty content corrupts the next turn's prefix; fail the conversation.
                if not generated_text:
                    raise ValueError(f"empty assistant content (turn {len(samples)})")

                prompt_token_ids = data.get("prompt_token_ids")
                completion_token_ids = choice.get("token_ids")
                if not prompt_token_ids or not completion_token_ids:
                    raise ValueError(
                        "endpoint returned no token ids; it must support "
                        "return_token_ids"
                    )

                input_ids, loss_mask = build_boundary_sample(
                    prompt_token_ids, completion_token_ids
                )
                # History keeps parsed content; the generated <think> is supervised
                # in this turn's completion tokens above.
                assistant_msg = {"role": "assistant", "content": generated_text}
                samples.append(
                    {
                        "id": f"{conv_id}_turn{len(samples)}",
                        # Conversation-level key for --resume; the row `id` is
                        # turn-suffixed and would never match a recomputed one.
                        "primary_id": conv_id,
                        "input_ids": input_ids,
                        "loss_mask": loss_mask,
                        # Review-only twin of input_ids; ignored by training.
                        "conversations": [*prefix, assistant_msg],
                        "metadata": {
                            "idx": idx,
                            "finish_reason": choice.get("finish_reason"),
                            "usage": data.get("usage") or {},
                            "endpoint": endpoint,
                            "sampling_params": args.sampling_params,
                        },
                    }
                )
                prefix.append(assistant_msg)

            # Written only after every turn succeeds, so any row in the output
            # file means the whole conversation is done (see load_seen).
            for sample in samples:
                out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["ok"] += 1
        except Exception as e:  # noqa: BLE001
            # Failures go to a separate error file, not the training output.
            error_output = {
                "id": conv_id,
                "metadata": {
                    "idx": idx,
                    "error": repr(e),
                    "turns_completed": len(samples),
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
    dataset_id = dataset_config.hf_path

    # Use dataset-specific defaults if not provided
    split = args.split if args.split is not None else dataset_config.split
    subset = args.subset if args.subset is not None else dataset_config.subset

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
    print(f"Prompt field: {dataset_config.prompt_field}")
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

                turns = prepare_row(row, dataset_config)
                if not turns:
                    continue

                primary_id = _primary_identifier(row)
                if primary_id in seen_ids:
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "primary_id": primary_id,
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
