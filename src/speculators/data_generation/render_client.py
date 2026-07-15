"""Client for vLLM's ``/v1/chat/completions/render`` endpoint.

Preprocessing derives every off-policy loss mask from render boundaries, and the
renders come from the same vLLM instance the pipeline already runs for
hidden-state extraction and serving -- one tokenizer for the mask, the hidden
states, and inference. Only ``token_ids`` are requested here: the mask is
computed from the boundary between two renders, not from the server's
``assistant_tokens_mask`` (which is only populated when the chat template
carries ``{% generation %}`` tags).
"""

from http import HTTPStatus

import httpx

from speculators.data_generation.vllm_client import InvalidResponseError, with_retries

DEFAULT_RENDER_TIMEOUT = 30


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

    No truncation is requested: boundary detection needs the full lengths, and
    over-length rows are clipped downstream once the boundary is known.
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions/render"

    body: dict = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        body["tools"] = tools
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs

    resp = httpx.post(url, json=body, timeout=timeout)

    if HTTPStatus.BAD_REQUEST <= resp.status_code < HTTPStatus.INTERNAL_SERVER_ERROR:
        # Deterministic client error (bad request, wrong URL) -- retrying wastes
        # requests without changing the outcome. InvalidResponseError short-
        # circuits @with_retries (see vllm_client._handle_retry_error).
        raise InvalidResponseError(
            f"Render endpoint returned {resp.status_code}: {resp.text[:500]}"
        )
    if resp.status_code != HTTPStatus.OK:
        raise RenderError(
            f"Render endpoint returned {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    if "token_ids" not in data:
        raise RenderError(
            f"Render endpoint response missing 'token_ids': {str(data)[:500]}"
        )
    return data["token_ids"]
