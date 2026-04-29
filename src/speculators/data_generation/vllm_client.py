import asyncio
import base64
import functools
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import openai

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 15  # seconds
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class InvalidResponseError(Exception):
    pass


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


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

    prepared = []
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


def _prepare_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert processor-style multimodal messages to vLLM chat messages."""
    prepared = []
    for message in messages:
        prepared_message: dict[str, Any] = {
            "role": message.get("role"),
            "content": _prepare_chat_message_content(message.get("content", "")),
        }
        if "name" in message:
            prepared_message["name"] = message["name"]
        prepared.append(prepared_message)
    return prepared


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


def extract_output(completion, token_ids) -> str:
    prompt_token_ids = _get_field(completion, "prompt_token_ids")
    if prompt_token_ids is None:
        choices = _get_field(completion, "choices")
        if choices:
            prompt_token_ids = _get_field(choices[0], "prompt_token_ids")

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    if prompt_token_ids != token_ids:
        raise InvalidResponseError(
            f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
        )

    kv_transfer_params = _get_field(completion, "kv_transfer_params")
    if kv_transfer_params is None:
        raise InvalidResponseError("Response missing kv_transfer_params")

    hidden_states_path = _get_field(kv_transfer_params, "hidden_states_path")
    if hidden_states_path is None:
        raise InvalidResponseError("Response missing hidden_states_path")
    return hidden_states_path


@with_retries
async def generate_hidden_states_async(
    client: openai.AsyncClient,
    model: str,
    token_ids: list[int],
    messages: list[dict[str, Any]] | None = None,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.

    Args:
        client: The async OpenAI client.
        model: The model ID.
        token_ids: The input token IDs.
        messages: Optional chat messages for vLLM multimodal requests.
        timeout: Timeout in seconds for each request attempt. None for no timeout.
    """
    if messages is not None:
        chat_messages = _prepare_chat_messages(messages)
        coro = client.chat.completions.create(
            model=model,
            messages=cast("Any", chat_messages),
            max_tokens=1,
            extra_body={"return_token_ids": True, "add_generation_prompt": False},
            timeout=timeout,
        )
    else:
        coro = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    if timeout is not None:
        completion = await asyncio.wait_for(coro, timeout=timeout)
    else:
        completion = await coro

    return extract_output(completion, token_ids)


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    token_ids: list[int],
    messages: list[dict[str, Any]] | None = None,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.
    """
    if messages is not None:
        chat_messages = _prepare_chat_messages(messages)
        completion = client.chat.completions.create(
            model=model,
            messages=cast("Any", chat_messages),
            max_tokens=1,
            extra_body={"return_token_ids": True, "add_generation_prompt": False},
            timeout=timeout,
        )
    else:
        completion = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    return extract_output(completion, token_ids)
