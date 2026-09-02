"""Regenerate-responses command — on-policy response regeneration via vLLM."""

import asyncio
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import aiohttp
import typer
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

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


# ---------------------------------------------------------------------------
# Row ingestion: user/system turns, tool schema, cached tool results
# ---------------------------------------------------------------------------


def _conversation_messages(row: dict[str, Any]) -> list:
    """The ``messages``, ``conversations``, or message-list ``prompt``, else []."""
    convs = row.get("messages")
    if not (isinstance(convs, list) and convs):
        convs = row.get("conversations")
    if not (isinstance(convs, list) and convs):
        convs = row.get("prompt")
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
    if isinstance(prompt, str) and prompt:
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
    if not Path(path).is_file():
        return seen

    with Path(path).open(encoding="utf-8") as f:
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
                typer.echo(f"Auto-detected model: {model_name}")
                return model_name
            raise ValueError("No models found at endpoint")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}\n"
            f"Please specify model with --model argument"
        ) from e


def build_detokenizer(model: str) -> Callable[[list[int]], str]:
    """Return a decoder for the review-only ``text`` twin (see _sample_from_response).

    Loads ``model``'s tokenizer so the twin is exactly ``decode(input_ids)``;
    ``skip_special_tokens=False`` keeps the chat/control tokens (``<|im_start|>``,
    ``<think>``, ``<|im_end|>``) visible. Pass a tokenizer path as ``--model``
    (the checkpoint, not a ``--served-model-name`` alias).
    """
    typer.echo(f"Loading tokenizer: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)

    def detokenize(token_ids: list[int]) -> str:
        # decode() is typed str | list[str]; a 1-D id list always yields str.
        return cast("str", tokenizer.decode(token_ids, skip_special_tokens=False))

    return detokenize


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
    detokenize: Callable[[list[int]], str],
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
        # Review-only decode of input_ids; ignored by training. Faithful to the
        # tokens by construction -- system/user context, the template's <think>
        # priming, and history with prior-turn reasoning already stripped.
        "text": detokenize(input_ids),
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
    detokenize: Callable[[list[int]], str],
    reasoning_effort: str | None = None,
    temperature: float | None = None,
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
            overrides = {
                k: v
                for k, v in (
                    ("reasoning_effort", reasoning_effort),
                    ("temperature", temperature),
                )
                if v is not None
            }
            recorded_params = {**sampling_params, **overrides}
            payload: dict[str, Any] = {
                # Spread first: the keys below are ours to own and must not be
                # overridden by user-supplied sampling params.
                **recorded_params,
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
                detokenize=detokenize,
                conv_id=conv_id,
                sample_index=len(samples),
                idx=item["idx"],
                endpoint=endpoint,
                sampling_params=recorded_params,
            )
            samples.append(sample)
            prefix.append(assistant_msg)

            prompt_token_ids = data.get("prompt_token_ids")
            completion_token_ids = data["choices"][0].get("token_ids")
            total_tokens = len(prompt_token_ids or []) + len(completion_token_ids or [])
            if total_tokens > max_tokens:
                truncated = True
                break

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
    typer.echo(
        f"\nPipeline complete in {elapsed:.1f}s: {n_convs} conversations "
        f"({stats['ok']} ok, {stats['errors']} errors, "
        f"{stats['truncated']} truncated)"
    )
    if stats["requests"] > 0:
        rps = stats["requests"] / elapsed if elapsed > 0 else 0
        tps = int(stats["completion_tokens"] / elapsed) if elapsed > 0 else 0
        avg_latency = stats["total_request_s"] / stats["requests"] * 1000
        typer.echo(
            f"Throughput: {rps:.1f} req/s, {tps} tok/s | "
            f"Avg request latency: {avg_latency:.0f} ms"
        )


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def ensure_parent_dirs(*paths: str) -> None:
    """Create parent directories for output files.

    open(..., "a") creates the file but not its parent directories.
    """
    for path in paths:
        parent = Path(path).parent
        if parent != Path():
            parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Worker pool & orchestration
# ---------------------------------------------------------------------------


async def _worker(
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    *,
    model: str,
    max_tokens: int,
    endpoint: str,
    sampling_params: dict[str, Any],
    max_retries: int,
    out_fh,
    err_fh,
    progress,
    stats: dict[str, Any],
    detokenize: Callable[[list[int]], str],
):
    """Pull conversations off the queue and regenerate them into boundary rows.

    Each target generation becomes one boundary sample; tool calls reuse the
    source data's cached results (see ``regenerate_conversation``). Truncated
    conversations still emit the rows completed before the cut.
    """

    async def post(payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = await _post_chat(session, endpoint, payload, max_retries=max_retries)
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

        conv_id = item["primary_id"]
        # Held by the caller so a mid-conversation failure can still report how
        # many rows had been completed.
        samples: list[dict[str, Any]] = []
        try:
            truncated = await regenerate_conversation(
                post,
                item,
                model=model,
                max_tokens=max_tokens,
                endpoint=endpoint,
                sampling_params=sampling_params,
                samples=samples,
                detokenize=detokenize,
                reasoning_effort=item.get("reasoning_effort"),
                temperature=item.get("temperature"),
            )
            # Written only after the conversation finishes -- a clean truncation
            # included, since rerunning it would truncate again. An exception
            # writes nothing, so any row in the output file means the
            # conversation needs no rerun (see load_seen).
            for sample in samples:
                out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                usage = sample.get("metadata", {}).get("usage", {})
                stats["completion_tokens"] += usage.get("completion_tokens", 0)
            out_fh.flush()
            if samples:
                stats["ok"] += 1
            if truncated or any(
                s.get("metadata", {}).get("finish_reason") == "length" for s in samples
            ):
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


def load_input_dataset(
    dataset_name: str, split: str | None, subset: str | None
) -> tuple[DatasetConfig, Any, str]:
    """Load either a registered Hugging Face preset or a local JSON/JSONL file."""
    if dataset_name not in REGEN_DATASETS:
        config = DatasetConfig(
            name=Path(dataset_name).stem,
            hf_path=dataset_name,
            split="train",
            prompt_field="prompt",
        )
        dataset = load_dataset(
            "json", data_files=config.hf_path, split=config.split, streaming=True
        )
        return config, dataset, config.split

    config = DATASET_CONFIGS[dataset_name]
    resolved_split = split if split is not None else config.split
    resolved_subset = subset if subset is not None else config.subset
    dataset = load_dataset(
        config.hf_path, name=resolved_subset, split=resolved_split, streaming=True
    )
    return config, dataset, resolved_split


async def _run(  # noqa: C901
    *,
    endpoint: str,
    model: str | None,
    dataset_name: str,
    split: str | None,
    subset: str | None,
    limit: int | None,
    concurrency: int,
    max_tokens: int,
    sampling_params: dict[str, Any],
    outfile: str | None,
    resume: bool,
    language_filter: str | None,
    max_retries: int,
    reasoning_effort_cycle: list[str] | None,
    temperature_cycle: list[float] | None,
    seed: int | None,
) -> None:
    """Main async function to process dataset through vLLM endpoints."""
    typer.echo(f"Using endpoint: {endpoint}")

    # Auto-detect model if not specified
    if model is None:
        model = await detect_model(endpoint)

    typer.echo(f"Using model: {model}")
    if reasoning_effort_cycle:
        typer.echo(
            "Reasoning effort (random per conversation): "
            f"{', '.join(reasoning_effort_cycle)}"
        )
    if temperature_cycle:
        typer.echo(
            "Temperature (random per conversation): "
            f"{', '.join(str(t) for t in temperature_cycle)}"
        )
    if reasoning_effort_cycle or temperature_cycle:
        typer.echo(f"Sampling seed: {seed}")

    # Decoder for the review-only `text` twin; see build_detokenizer.
    detokenize = build_detokenizer(model)

    dataset_config, hf_dataset, split = load_input_dataset(dataset_name, split, subset)
    dataset_id = dataset_config.hf_path

    # Generate output filename if not specified
    if outfile is None:
        # Extract simple model name from full path
        model_short = model.split("/")[-1] if "/" in model else model
        model_short = sanitize_filename(model_short)
        source_name = sanitize_filename(dataset_config.name)
        outfile = f"{source_name}_{model_short}.jsonl"

    # Failed / partial conversations are written here instead of the training file.
    outpath = Path(outfile)
    suffix = outpath.suffix or ".jsonl"
    error_outfile = str(outpath.with_name(f"{outpath.stem}.errors{suffix}"))

    typer.echo(f"Using dataset: {dataset_id}")
    typer.echo(f"Split: {split}")
    typer.echo(f"Prompt field: {dataset_config.prompt_field}")
    typer.echo(f"Output file: {outfile}")
    typer.echo(f"Error file: {error_outfile}")
    typer.echo()

    seen_ids = load_seen(outfile) if resume else set()

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)

    ensure_parent_dirs(outfile, error_outfile)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    connector = aiohttp.TCPConnector(
        limit=0, force_close=False, enable_cleanup_closed=True
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        with (
            Path(outfile).open("a", encoding="utf-8") as output_file,  # noqa: ASYNC230
            Path(error_outfile).open("a", encoding="utf-8") as error_file,  # noqa: ASYNC230
            tqdm(
                total=limit,
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
                "completion_tokens": 0,
                "total_request_s": 0.0,
                "start_time": time.perf_counter(),
            }
            workers = [
                asyncio.create_task(
                    _worker(
                        session,
                        queue,
                        model=model,
                        max_tokens=max_tokens,
                        endpoint=endpoint,
                        sampling_params=sampling_params,
                        max_retries=max_retries,
                        out_fh=output_file,
                        err_fh=error_file,
                        progress=progress,
                        stats=stats,
                        detokenize=detokenize,
                    )
                )
                for _ in range(concurrency)
            ]

            rng = random.Random(seed)
            processed_count = 0
            for index, row in enumerate(hf_dataset):
                if limit is not None and processed_count >= limit:
                    break

                if language_filter and row.get("language") != language_filter:
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

                queue_item: dict[str, Any] = {
                    "idx": index,
                    "primary_id": primary_id,
                    "turns": turns,
                    "tools": tools,
                    "tool_results": tool_results,
                }
                # Independently pick a reasoning effort and temperature at
                # random so that, across conversations, we sample a mix of
                # (effort, temperature) combinations rather than always pairing
                # effort[i] with temperature[i]. Seeded via --seed for
                # reproducibility.
                if reasoning_effort_cycle is not None:
                    queue_item["reasoning_effort"] = rng.choice(reasoning_effort_cycle)
                if temperature_cycle is not None:
                    queue_item["temperature"] = rng.choice(temperature_cycle)
                await queue.put(queue_item)
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)

            _log_summary(stats)


def _validate_dataset(value: str) -> str:
    """Validate a registered preset or a local JSON/JSONL file."""
    if value in MULTIMODAL_DATASETS:
        raise typer.BadParameter(
            f"{value!r} is multimodal; on-policy regeneration does not support "
            "images yet. Use it off-policy with `prepare-data`."
        )
    if value in REGEN_DATASETS:
        return value

    dataset_path = Path(value)
    if dataset_path.suffix.lower() not in {".json", ".jsonl"}:
        raise typer.BadParameter(
            f"{value!r} is not a supported dataset preset or local .json/.jsonl file"
        )
    if not dataset_path.is_file():
        raise typer.BadParameter(f"dataset file does not exist: {value}")
    return str(dataset_path)


def regenerate_responses(  # noqa: C901
    endpoint: Annotated[
        str,
        typer.Option(
            help="vLLM OpenAI-compatible Chat Completions endpoint",
        ),
    ] = "http://127.0.0.1:8000/v1/chat/completions",
    model: Annotated[
        str | None,
        typer.Option(
            help="Model name exposed by vLLM (auto-detected if not specified)",
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option(
            help=(
                "Registered dataset preset or local .json/.jsonl file. "
                "Presets: " + ", ".join(REGEN_DATASETS)
            ),
            metavar="PRESET_OR_PATH",
            callback=_validate_dataset,
        ),
    ] = "ultrachat",
    split: Annotated[
        str | None,
        typer.Option(
            help="Dataset split (defaults to dataset-specific split)",
        ),
    ] = None,
    subset: Annotated[
        str | None,
        typer.Option(
            help=(
                "Dataset subset/config name "
                "(auto-detected from dataset config if not specified)"
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Stop after N rows"),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(help="Max concurrent requests"),
    ] = 64,
    max_tokens: Annotated[
        int,
        typer.Option(help="max_tokens for generation"),
    ] = 8192,
    sampling_params: Annotated[
        str | None,
        typer.Option(
            help=(
                "JSON object merged into each chat-completion request, "
                'e.g. \'{"temperature": 0.6, "top_p": 0.95, "seed": 0}\''
            ),
        ),
    ] = None,
    outfile: Annotated[
        str | None,
        typer.Option(
            help="Output JSONL path (auto-generated if not specified)",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip rows already in outfile (by stable primary id)",
        ),
    ] = False,
    language_filter: Annotated[
        str | None,
        typer.Option(
            help="Only process rows where language==this (e.g., EN)",
        ),
    ] = None,
    max_retries: Annotated[
        int,
        typer.Option(
            help=(
                "Max retry attempts per request on transient failure "
                f"(default: {DEFAULT_MAX_RETRIES})"
            ),
        ),
    ] = DEFAULT_MAX_RETRIES,
    reasoning_effort_cycle: Annotated[
        str | None,
        typer.Option(
            help=(
                "Comma-separated reasoning effort levels to cycle through "
                "per conversation, e.g. 'low,high,max'"
            ),
        ),
    ] = None,
    temperature_cycle: Annotated[
        str | None,
        typer.Option(
            help=(
                "Comma-separated temperature values to cycle through "
                "per conversation, e.g. '0.6,0.8,1.0'"
            ),
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help=(
                "Seed for random selection of reasoning effort / temperature "
                "per conversation, for reproducible runs"
            ),
        ),
    ] = None,
) -> None:
    """Regenerate dataset responses via a vLLM Chat API endpoint.

    Connects to a running vLLM instance, regenerates assistant responses for a
    dataset's conversations on-policy, and writes boundary training samples
    (prompt + completion token ids with loss mask) to a JSONL file.
    """
    if dataset not in REGEN_DATASETS and (split is not None or subset is not None):
        raise typer.BadParameter("--split and --subset only apply to dataset presets")

    if max_retries < 0:
        raise typer.BadParameter("--max-retries must be >= 0")

    parsed_sampling_params: dict[str, Any] = {}
    if sampling_params is not None:
        try:
            parsed_sampling_params = json.loads(sampling_params)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"--sampling-params is not valid JSON: {e}") from e
        if not isinstance(parsed_sampling_params, dict):
            raise typer.BadParameter("--sampling-params must be a JSON object")

    parsed_reasoning_effort_cycle: list[str] | None = None
    if reasoning_effort_cycle is not None:
        parsed_reasoning_effort_cycle = [
            level.strip() for level in reasoning_effort_cycle.split(",")
        ]
        if not parsed_reasoning_effort_cycle:
            raise typer.BadParameter("--reasoning-effort-cycle must not be empty")

    parsed_temperature_cycle: list[float] | None = None
    if temperature_cycle is not None:
        try:
            parsed_temperature_cycle = [
                float(t.strip()) for t in temperature_cycle.split(",")
            ]
        except ValueError as e:
            raise typer.BadParameter(
                "--temperature-cycle values must be numbers"
            ) from e
        if not parsed_temperature_cycle:
            raise typer.BadParameter("--temperature-cycle must not be empty")

    try:
        asyncio.run(
            _run(
                endpoint=endpoint,
                model=model,
                dataset_name=dataset,
                split=split,
                subset=subset,
                limit=limit,
                concurrency=concurrency,
                max_tokens=max_tokens,
                sampling_params=parsed_sampling_params,
                outfile=outfile,
                resume=resume,
                language_filter=language_filter,
                max_retries=max_retries,
                reasoning_effort_cycle=parsed_reasoning_effort_cycle,
                temperature_cycle=parsed_temperature_cycle,
                seed=seed,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)
