"""Client for vLLM's /v1/chat/completions/render endpoint (RFC #652)."""

from http import HTTPStatus

import httpx

from speculators.data_generation.vllm_client import with_retries

DEFAULT_RENDER_TIMEOUT = 30


class RenderError(Exception):
    """Non-200 response from the render endpoint."""


@with_retries
def render_conversation(
    endpoint: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
    max_length: int | None = None,
    timeout: float = DEFAULT_RENDER_TIMEOUT,
) -> dict:
    """POST to /v1/chat/completions/render and return token_ids + loss_mask."""
    url = f"{endpoint.rstrip('/')}/v1/chat/completions/render"

    body: dict = {
        "messages": messages,
        "add_generation_prompt": False,
    }
    if tools is not None:
        body["tools"] = tools
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if max_length is not None:
        body["max_tokens"] = max_length

    resp = httpx.post(url, json=body, timeout=timeout)

    if resp.status_code != HTTPStatus.OK:
        raise RenderError(
            f"Render endpoint returned {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    return {"token_ids": data["token_ids"], "loss_mask": data.get("loss_mask")}
