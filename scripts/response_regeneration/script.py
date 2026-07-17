#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import deque
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

logger = logging.getLogger(__name__)

# On-policy regeneration has no multimodal support yet; off-policy `prepare-data`
# does, so these presets are gated here rather than dropped from the registry.
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


# ---------------------------------------------------------------------------
# CLI & run configuration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Row ingestion: user/system turns, tool schema, cached tool results
# ---------------------------------------------------------------------------


def _conversation_messages(row: dict[str, Any]) -> list:
    """The ``messages`` or ``conversations`` list from a row, else []."""
    convs = row.get("messages")
    if not (isinstance(convs, list) and convs):
        convs = row.get("conversations")
    return convs if isinstance(convs, list) else []


def _message_role_content(m: dict) -> tuple[str | None, Any]:
    """Canonical ``(role, content)`` for a message across the role/content and
    from/value schemas. ``role`` collapses ``human`` to ``user``; ``system`` and
    ``tool`` pass through; anything else (assistant/gpt) returns ``None``."""
    role = m.get("role") or m.get("from")
    content = m.get("content")
    if content is None:
        content = m.get("value")
    if role in ("user", "human"):
        return "user", content
    if role in ("system", "tool"):
        return role, content
    return None, content


def extract_conversation(
    row: dict[str, Any], prompt_field: str | None
) -> tuple[list[dict[str, Any]], list[tuple[Any, list[str]]]]:
    """Read the regeneration turns and the cached tool results in one pass.

    Walks a ``messages``/``conversations`` field (role/content or from/value
    schema). System and user turns drive regeneration; the original assistant
    turns are dropped and regenerated. Each tool-result turn is captured as a
    ``(content, tool_names)`` pair, where ``tool_names`` are the tools that
    result answers (read from its ``<tool_response>`` payload) -- used to guard
    the positional splice against a call for a different tool. Rows without a
    usable conversation fall back to a single ``prompt_field`` user turn and
    carry no results.
    """
    turns: list[dict[str, Any]] = []
    results: list[tuple[Any, list[str]]] = []
    for m in _conversation_messages(row):
        if not isinstance(m, dict):
            continue
        role, content = _message_role_content(m)
        if role in ("system", "user") and content:
            turns.append({"role": role, "content": content})
        elif role == "tool" and content is not None:
            results.append((content, _tool_result_names(content)))
        # original assistant/gpt turns are dropped and regenerated
    if any(turn["role"] == "user" for turn in turns):
        return turns, results

    # no usable user turn: fall back to the prompt_field
    prompt = row.get(prompt_field) if prompt_field else None
    if prompt:
        return [{"role": "user", "content": prompt}], []
    return [], []


def prepare_row(
    row: dict[str, Any], config: DatasetConfig
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[Any, list[str]]]] | None:
    """The normalized row, its regeneration turns, and cached tool results.

    ``filter_fn`` sees the raw row; ``normalize_fn`` is merged over it (HF
    ``map`` keeps raw columns). Turns and results are read from that one
    normalized row so they stay paired; ``None`` to skip the row.
    """
    if config.filter_fn is not None and not config.filter_fn(row):
        return None
    normalized = {**row, **config.normalize_fn(row)} if config.normalize_fn else row
    turns, tool_results = extract_conversation(normalized, config.prompt_field)
    if not turns:
        return None
    return normalized, turns, tool_results


def _maybe_json(value: Any):
    """Best-effort JSON decode; return None if the value is not valid JSON."""
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _list_field(row, key) -> list | None:
    """Non-empty list from ``row[key]`` (JSON-decoded if a string), else None."""
    value = row.get(key)
    if isinstance(value, str):
        value = _maybe_json(value)
    return value if isinstance(value, list) and value else None


def extract_tools(row) -> list | None:
    """Return the OpenAI-style ``tools`` schema for a row, or ``None``.

    Reads the ``tools`` column -- a list, or a JSON-string encoding one, as the
    Hermes function-calling dataset stores it. A row that declares a ``tools``
    field we cannot read as a list raises ``ValueError`` rather than silently
    regenerating tool-free; a tool-free row returns ``None``.
    """
    tools = _list_field(row, "tools")
    if tools:
        return tools
    if row.get("tools") not in (None, "", [], {}):
        raise ValueError("a tools field is present but not a usable list")
    return None


_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _tool_result_names(content: Any) -> list[str]:
    """Tool names a cached result answers, for the splice name-match guard.

    Hermes embeds the answering tool's name in each ``<tool_response>`` payload.
    Returns ``[]`` when no name can be read, which disables the guard for that
    result rather than blocking the splice on our own parse miss.
    """
    if not isinstance(content, str):
        return []
    names = []
    for block in _TOOL_RESPONSE_RE.findall(content):
        obj = _maybe_json(block)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            names.append(obj["name"])
    return names


# ---------------------------------------------------------------------------
# Resume state & vLLM server IO
# ---------------------------------------------------------------------------


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

    A conversation fans out to one row per target generation -- a tool call or a
    final answer -- whose ``id`` carries a ``_gen<N>`` suffix; the conversation's
    own :func:`_primary_identifier` is kept alongside it as ``primary_id``.
    Resume keys on that, since the suffixed ids never match a recomputed one.
    Rows are written only after the conversation finishes, so one row is enough
    to mark it done.

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


# ---------------------------------------------------------------------------
# Regeneration: model response -> boundary training samples
# ---------------------------------------------------------------------------


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


def _tool_result_message(tool_call: dict, content: str) -> dict[str, Any]:
    """Build the ``tool`` message that feeds a cached (off-policy) result back to
    the target, paired to the id of the call the target just generated."""
    message: dict[str, Any] = {"role": "tool", "content": content}
    call_id = tool_call.get("id")
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _sample_from_response(
    data: dict[str, Any],
    *,
    prefix: list[dict[str, Any]],
    conv_id: str,
    sample_index: int,
    idx: int,
    endpoint: str,
    sampling_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list | None]:
    """Turn one chat-completion response into a boundary sample and the assistant
    message to append to the running prefix.

    Returns ``(sample, assistant_msg, tool_calls)``; ``tool_calls`` is truthy when
    the target emitted a tool call (its content may be empty). Raises on a wholly
    empty generation or a response missing the ``return_token_ids`` payload.
    """
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content")
    tool_calls = message.get("tool_calls")

    # A tool call legitimately has empty content; only a wholly empty generation
    # corrupts the next prefix and must fail the conversation.
    if not content and not tool_calls:
        raise ValueError(f"empty assistant generation (sample {sample_index})")

    prompt_token_ids = data.get("prompt_token_ids")
    completion_token_ids = choice.get("token_ids")
    if not prompt_token_ids or not completion_token_ids:
        raise ValueError(
            "endpoint returned no token ids; it must support return_token_ids"
        )

    input_ids, loss_mask = build_boundary_sample(prompt_token_ids, completion_token_ids)
    if tool_calls:
        # History keeps the parsed call; any generated <think> is supervised in
        # this row's completion tokens, not re-rendered.
        assistant_msg = {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls,
        }
    else:
        assistant_msg = {"role": "assistant", "content": content}

    sample = {
        "id": f"{conv_id}_gen{sample_index}",
        # Conversation-level key for --resume; the row `id` is generation-suffixed
        # and would never match a recomputed one.
        "primary_id": conv_id,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        # Review-only twin of input_ids; ignored by training.
        "conversations": [*prefix, assistant_msg],
        "metadata": {
            "idx": idx,
            "finish_reason": choice.get("finish_reason"),
            "is_tool_call": bool(tool_calls),
            "usage": data.get("usage") or {},
            "endpoint": endpoint,
            "sampling_params": sampling_params,
        },
    }
    return sample, assistant_msg, tool_calls


async def regenerate_conversation(
    post_fn,
    item: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    endpoint: str,
    sampling_params: dict[str, Any],
    samples: list[dict[str, Any]],
) -> bool:
    """Regenerate one conversation into per-generation boundary samples.

    Each target generation -- a tool call *or* a final answer -- is one boundary
    row (loss_mask 0 over the prompt, 1 over the generated tokens). Tool calls
    are not executed: the target's i-th regenerated call is paired with the i-th
    cached result from the source row.

    Returns whether the conversation was truncated -- which happens when a call
    cannot be paired 1:1 with a cached result (results exhausted, a parallel
    call, or a different tool than the result answers). Completed rows are
    appended to ``samples`` as they go, so the caller keeps partial progress if
    this raises.
    """
    turns = item["turns"]
    tools = item.get("tools")
    tool_results = deque(item.get("tool_results") or [])
    conv_id = item["primary_id"]

    prefix: list[dict[str, Any]] = []
    truncated = False

    for turn in turns:
        if turn["role"] == "system":
            prefix.append({"role": "system", "content": turn["content"]})
            continue

        prefix.append({"role": "user", "content": turn["content"]})

        # Tool-call loop: a tool call splices a cached result and continues;
        # a final answer ends the turn.
        while True:
            payload: dict[str, Any] = {
                # Spread first: the keys below are ours to own and must not be
                # overridden by user-supplied sampling params.
                **sampling_params,
                "model": model,
                "messages": prefix,
                "max_tokens": max_tokens,
                "return_token_ids": True,  # prompt_token_ids + completion token_ids
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            data = await post_fn(payload)
            sample, assistant_msg, tool_calls = _sample_from_response(
                data,
                prefix=prefix,
                conv_id=conv_id,
                sample_index=len(samples),
                idx=item["idx"],
                endpoint=endpoint,
                sampling_params=sampling_params,
            )
            samples.append(sample)
            prefix.append(assistant_msg)

            if not tool_calls:
                break  # final answer: this user turn is done

            # To continue past a tool call we need exactly one call and a cached
            # result to pair with it; otherwise keep this (committed) call row
            # and truncate.
            # TODO(regen): support parallel/multi tool calls -- a turn with >1
            # call truncates here (dropped ~16/50 rows in a Hermes validation run).
            if len(tool_calls) != 1 or not tool_results:
                truncated = True
                break

            content, result_names = tool_results.popleft()
            call_name = (tool_calls[0].get("function") or {}).get("name")
            # Splice only if the regenerated call is for the tool this cached
            # result answers; a different tool cannot be paired coherently, so
            # keep the committed call row and truncate.
            if result_names and call_name not in result_names:
                truncated = True
                break

            prefix.append(_tool_result_message(tool_calls[0], content))

        if truncated:
            break

    return truncated


def _log_summary(stats: dict[str, Any]) -> None:
    elapsed = time.perf_counter() - stats["start_time"]
    n_convs = stats["ok"] + stats["errors"]
    logger.info(
        "Pipeline complete in %.1fs: %d conversations (%d ok, %d errors, %d truncated)",
        elapsed,
        n_convs,
        stats["ok"],
        stats["errors"],
        stats["truncated"],
    )
    if stats["requests"] > 0:
        logger.info(
            "Throughput: %.1f req/s, %d tok/s | "
            "Avg request latency: %.0f ms | "
            "Avg queue wait: %.0f ms",
            stats["requests"] / elapsed if elapsed > 0 else 0,
            int(stats["completion_tokens"] / elapsed) if elapsed > 0 else 0,
            stats["total_request_s"] / stats["requests"] * 1000,
            stats["total_queue_wait_s"] / n_convs * 1000 if n_convs > 0 else 0,
        )


# ---------------------------------------------------------------------------
# Worker pool & orchestration
# ---------------------------------------------------------------------------


async def worker(
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    args,
    out_fh,
    err_fh,
    endpoint: str,
    progress,
    stats: dict[str, Any],
):
    """Pull conversations off the queue and regenerate them into boundary rows.

    Each target generation becomes one boundary sample; tool calls reuse the
    source data's cached results (see ``regenerate_conversation``). Truncated
    conversations still emit the rows completed before the cut.
    """

    async def post(payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = await _post_chat(
            session, endpoint, payload, max_retries=args.max_retries
        )
        latency = time.perf_counter() - t0
        stats["total_request_s"] += latency
        stats["requests"] += 1
        logger.debug("vLLM request completed in %.0f ms", latency * 1000)
        return result

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        enqueued_at = item.pop("enqueued_at", None)
        if enqueued_at is not None:
            stats["total_queue_wait_s"] += time.perf_counter() - enqueued_at

        conv_id = item["primary_id"]
        # Held by the caller so a mid-conversation failure can still report how
        # many rows had been completed.
        samples: list[dict[str, Any]] = []
        try:
            truncated = await regenerate_conversation(
                post,
                item,
                model=args.model,
                max_tokens=args.max_tokens,
                endpoint=endpoint,
                sampling_params=args.sampling_params,
                samples=samples,
            )
            # Written only after the conversation finishes -- a clean truncation
            # included, since rerunning it would truncate again. An exception
            # writes nothing, so any row in the output file means the
            # conversation needs no rerun (see load_seen).
            for sample in samples:
                out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                usage = sample.get("metadata", {}).get("usage", {})
                stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                stats["completion_tokens"] += usage.get("completion_tokens", 0)
            out_fh.flush()
            if samples:
                stats["ok"] += 1
            if truncated:
                stats["truncated"] += 1
        except Exception as e:  # noqa: BLE001
            # Failures go to a separate error file, not the training output.
            error_output = {
                "id": conv_id,
                "metadata": {
                    "idx": item["idx"],
                    "error": repr(e),
                    "generations_completed": len(samples),
                    "endpoint": endpoint,
                },
            }
            err_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            err_fh.flush()
            stats["errors"] += 1
        finally:
            elapsed = time.perf_counter() - stats["start_time"]
            postfix = {
                "ok": stats["ok"],
                "err": stats["errors"],
                "trunc": stats["truncated"],
            }
            if elapsed > 0 and stats["requests"] > 0:
                postfix["rps"] = f"{stats['requests'] / elapsed:.1f}"
                postfix["tps"] = f"{stats['completion_tokens'] / elapsed:.0f}"
            progress.set_postfix(postfix, refresh=False)
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
            stats = {
                "ok": 0,
                "errors": 0,
                "truncated": 0,
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_request_s": 0.0,
                "total_queue_wait_s": 0.0,
                "start_time": time.perf_counter(),
            }
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

                prepared = prepare_row(row, dataset_config)
                if prepared is None:
                    continue
                normalized, turns, tool_results = prepared

                primary_id = _primary_identifier(row)
                if primary_id in seen_ids:
                    continue

                # Broken input tool schema: record and skip (don't crash the run).
                try:
                    tools = extract_tools(normalized)
                except ValueError as exc:
                    logger.warning(
                        "Skipping row %s: input tool schema is broken (%s)",
                        primary_id,
                        exc,
                    )
                    error_output = {
                        "id": primary_id,
                        "metadata": {
                            "idx": index,
                            "error": repr(exc),
                            "generations_completed": 0,
                            "endpoint": endpoint,
                        },
                    }
                    error_file.write(
                        json.dumps(error_output, ensure_ascii=False) + "\n"
                    )
                    error_file.flush()
                    stats["errors"] += 1
                    progress.update(1)
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "primary_id": primary_id,
                        "turns": turns,
                        "tools": tools,
                        "tool_results": tool_results,
                        "enqueued_at": time.perf_counter(),
                    }
                )
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)

            _log_summary(stats)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
