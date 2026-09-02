"""Client for vLLM's ``/v1/chat/completions/render`` endpoint.

Only ``token_ids`` are requested: the loss mask is derived from the boundary
between two renders, not the server's ``assistant_tokens_mask`` (which needs
``{% generation %}`` tags). Renders come from the vLLM instance the pipeline
already runs, so one tokenizer feeds the mask, hidden states, and serving.
"""

import logging
import os
from http import HTTPStatus

import httpx

from speculators.data_generation.vllm_client import InvalidResponseError, with_retries

logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_RENDER_TIMEOUT = 30

_client: httpx.Client | None = None
_client_pid: int | None = None


def _post(url: str, *, json: dict, timeout: float) -> httpx.Response:
    """POST through a per-process pooled client.

    Boundary derivation issues a couple of renders per assistant turn, tens of
    thousands per dataset. Module-level ``httpx.post`` builds a new client
    (SSL context, CA bundle) and a new TCP connection for every call, ~2-3 ms
    of overhead each plus one throwaway socket. A pooled client removes both.

    The client is keyed by PID because ``datasets.map`` workers are forked
    processes: pooled sockets inherited across a fork would be shared with the
    parent and corrupt each other's responses.
    """
    global _client, _client_pid  # noqa: PLW0603
    pid = os.getpid()
    if _client is None or _client_pid != pid:
        _client = httpx.Client()
        _client_pid = pid
    return _client.post(url, json=json, timeout=timeout)


# 4xx that mean "retry", not "your request is wrong".
TRANSIENT_STATUSES = frozenset(
    {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS}
)


class RenderError(Exception):
    """Non-200, retry-eligible response from the render endpoint."""


@with_retries
def render_conversation(
    endpoint: str,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
    timeout: float = DEFAULT_RENDER_TIMEOUT,
) -> list[int]:
    """POST to ``/v1/chat/completions/render`` and return the token ids.

    No truncation: boundary detection needs full lengths; over-length rows are
    clipped downstream.
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions/render"

    body = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        body["tools"] = tools
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs

    resp = _post(url, json=body, timeout=timeout)

    if (
        HTTPStatus.BAD_REQUEST <= resp.status_code < HTTPStatus.INTERNAL_SERVER_ERROR
        and resp.status_code not in TRANSIENT_STATUSES
    ):
        # Deterministic client error (bad request, wrong URL) -- retrying wastes
        # requests without changing the outcome. InvalidResponseError short-
        # circuits @with_retries (see vllm_client._handle_retry_error).
        raise InvalidResponseError(
            f"Render endpoint returned {resp.status_code}: {resp.text}"
        )
    if resp.status_code != HTTPStatus.OK:
        raise RenderError(f"Render endpoint returned {resp.status_code}: {resp.text}")

    data = resp.json()
    if "token_ids" not in data:
        raise RenderError(f"Render endpoint response missing 'token_ids': {data}")
    return data["token_ids"]
