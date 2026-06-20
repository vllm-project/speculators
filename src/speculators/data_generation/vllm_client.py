import asyncio
import base64
import fcntl
import functools
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast
from urllib.parse import urlparse

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.completion import Completion
from safetensors.torch import load_file, save_file
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 120  # seconds
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class InvalidResponseError(Exception):
    pass


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_token_id_list(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return list(token_ids)


def _image_ref_to_chat_url(image_ref: Any) -> str:
    """Convert a dataset image reference to an OpenAI-compatible image URL."""
    ref = str(image_ref)
    parsed = urlparse(ref)
    if parsed.scheme in {"http", "https", "data", "file"}:
        return ref

    path = Path(ref).expanduser()
    if path.exists() and path.is_file():
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    if path.is_absolute():
        return path.as_uri()

    return ref


def _get_image_ref(part: dict[str, Any]) -> Any | None:
    if part.get("type") not in ("image", "image_url", "input_image"):
        return None

    image_ref = part.get("image")
    if image_ref is not None:
        return image_ref

    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    return image_url


def _prepare_chat_message_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content

    prepared: list[Any] = []
    for part in content:
        if isinstance(part, str):
            prepared.append({"type": "text", "text": part})
            continue

        if not isinstance(part, dict):
            prepared.append(part)
            continue

        image_ref = _get_image_ref(part)
        if image_ref is not None:
            prepared.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_ref_to_chat_url(image_ref)},
                }
            )
            continue

        text = part.get("text")
        if text is not None:
            prepared.append({"type": "text", "text": str(text)})
            continue

        prepared.append(part)

    return prepared


def _prepare_chat_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    """Convert processor-style multimodal messages to vLLM chat messages."""
    prepared = []
    for message in messages:
        prepared_message = dict(cast("dict[str, Any]", message))
        prepared_message["content"] = _prepare_chat_message_content(
            prepared_message.get("content", "")
        )
        prepared.append(prepared_message)
    return prepared


def _truncate_hidden_states_file(hidden_states_path: str, token_ids: list[int]) -> str:
    """Trim vLLM's full multimodal prompt output to the preprocessed prefix."""
    tensors = load_file(hidden_states_path)

    try:
        file_token_ids = _to_token_id_list(tensors["token_ids"])
        hidden_states = tensors["hidden_states"]
    except KeyError as exc:
        raise InvalidResponseError(
            f"Hidden states file missing {exc.args[0]}: {hidden_states_path}"
        ) from exc

    expected_len = len(token_ids)
    if (
        file_token_ids[:expected_len] != token_ids
        or hidden_states.shape[0] < expected_len
    ):
        raise InvalidResponseError(
            "Hidden states file does not match preprocessed prompt prefix: "
            f"expected {token_ids}, got {file_token_ids}"
        )

    # Safe for causal models: prefix hidden states cannot attend to future tokens
    # that only exist in vLLM's full chat-rendered prompt.
    tensors["token_ids"] = tensors["token_ids"][:expected_len].contiguous()
    tensors["hidden_states"] = hidden_states[:expected_len].contiguous()
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        hidden_states_path,
    )
    return hidden_states_path


def _handle_retry_error(
    error: Exception, attempt: int, total_attempts: int
) -> float | None:
    """Handle a retry-eligible error.

    Returns backoff seconds if the caller should retry, or ``None`` on the
    final attempt.  Raises ``InvalidResponseError`` immediately.
    """
    if isinstance(error, InvalidResponseError):
        raise error
    if attempt < total_attempts:
        backoff = RETRY_BACKOFF_BASE**attempt
        logger.warning(
            "Request aborted (attempt %d/%d): %s. Retrying in %ds...",
            attempt,
            total_attempts,
            error,
            backoff,
        )
        return backoff
    logger.error("Request timed out after %d attempts: %s", total_attempts, error)
    return None


def with_retries(fn):
    """Decorator that adds retry logic with exponential backoff.

    The decorated function gains a ``max_retries`` keyword argument
    (default ``DEFAULT_MAX_RETRIES``). ``InvalidResponseError`` is never
    retried. Works for both sync and async functions.
    """
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
            total_attempts = max_retries + 1
            last_error: Exception | None = None
            for attempt in range(1, total_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    backoff = _handle_retry_error(e, attempt, total_attempts)
                    if backoff is not None:
                        await asyncio.sleep(backoff)
            raise last_error  # type: ignore[misc]

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
        total_attempts = max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                backoff = _handle_retry_error(e, attempt, total_attempts)
                if backoff is not None:
                    time.sleep(backoff)
        raise last_error  # type: ignore[misc]

    return sync_wrapper


def extract_output(
    response: Completion | ChatCompletion,
    token_ids: list[int],
    *,
    allow_prefix_truncation: bool = False,
) -> str:
    token_ids = _to_token_id_list(token_ids)
    prompt_token_ids = _get_field(response, "prompt_token_ids")
    if prompt_token_ids is None:
        choices = _get_field(response, "choices")
        if choices:
            prompt_token_ids = _get_field(choices[0], "prompt_token_ids")

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    kv_transfer_params = _get_field(response, "kv_transfer_params")
    if kv_transfer_params is None:
        raise InvalidResponseError("Response missing kv_transfer_params")

    hidden_states_path = _get_field(kv_transfer_params, "hidden_states_path")
    if hidden_states_path is None:
        raise InvalidResponseError("Response missing hidden_states_path")

    prompt_token_ids = _to_token_id_list(prompt_token_ids)
    if prompt_token_ids == token_ids:
        return hidden_states_path

    if allow_prefix_truncation and prompt_token_ids[: len(token_ids)] == token_ids:
        logger.debug(
            "vLLM returned %d prompt tokens for a %d-token preprocessed prompt; "
            "truncating hidden states to match prepare_data output.",
            len(prompt_token_ids),
            len(token_ids),
        )
        return _truncate_hidden_states_file(hidden_states_path, token_ids)

    raise InvalidResponseError(
        f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
    )


class ClientItem(TypedDict):
    input_ids: list[int]
    """The input token IDs."""

    messages: NotRequired[list[ChatCompletionMessageParam]]
    """If provided, pass `messages` to Chat Completions API
    instead of passing `token_ids` to Completions API."""


async def _poll_lock_async(fd, poll_interval):
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            await asyncio.sleep(poll_interval)


async def wait_for_lock_async(lock_path, timeout=10.0, poll_interval=0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        await asyncio.wait_for(_poll_lock_async(fd, poll_interval), timeout=timeout)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


def wait_for_lock(lock_path, timeout=10.0, poll_interval=0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


@with_retries
async def generate_hidden_states_async(
    client: openai.AsyncClient,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.

    Args:
        client: The async OpenAI client.
        model: The model ID.
        client_item: Inputs to send via the client.
        timeout: Timeout in seconds for each request attempt. None for no timeout.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    coro: Coroutine[Any, Any, Completion | ChatCompletion]
    if messages is None:
        coro = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        chat_messages = _prepare_chat_messages(messages)
        coro = client.chat.completions.create(
            model=model,
            messages=cast("Any", chat_messages),
            max_tokens=1,
            extra_body={
                "add_generation_prompt": False,
                "continue_final_message": True,
                "return_token_ids": True,
            },
            timeout=timeout,
        )

    res: Completion | ChatCompletion
    if timeout is not None:
        res = await asyncio.wait_for(coro, timeout=timeout)
    else:
        res = await coro

    return extract_output(
        res,
        token_ids,
        allow_prefix_truncation=messages is not None,
    )


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    res: Completion | ChatCompletion
    if messages is None:
        res = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        chat_messages = _prepare_chat_messages(messages)
        res = client.chat.completions.create(
            model=model,
            messages=cast("Any", chat_messages),
            max_tokens=1,
            extra_body={
                "add_generation_prompt": False,
                "continue_final_message": True,
                "return_token_ids": True,
            },
            timeout=timeout,
        )

    return extract_output(
        res,
        token_ids,
        allow_prefix_truncation=messages is not None,
    )
