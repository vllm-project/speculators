"""Tests for render_client.py."""

import json

import httpx
import pytest

import speculators.data_generation.render_client as render_mod
from speculators.data_generation.render_client import (
    RenderError,
    render_conversation,
)

MOCK_TOKEN_IDS = [
    151644,
    872,
    198,
    9707,
    151645,
    198,
    151644,
    77091,
    198,
    13048,
    0,
    151645,
]

MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]


def _make_transport(*, token_ids=None, assistant_tokens_mask=None, status_code=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status_code != 200:
            return httpx.Response(status_code, text="error from mock")
        body = {"token_ids": token_ids or MOCK_TOKEN_IDS}
        if assistant_tokens_mask is not None:
            body["assistant_tokens_mask"] = assistant_tokens_mask
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _capturing_transport():
    """Return (transport, captured_body_dict)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"token_ids": MOCK_TOKEN_IDS})

    return httpx.MockTransport(handler), captured


def _call_render(transport, **kwargs):
    original_post = httpx.post

    def mock_post(url, **kw):
        return httpx.Client(transport=transport).request("POST", url, **kw)

    render_mod.httpx.post = mock_post  # type: ignore[attr-defined]
    try:
        return render_conversation(
            "http://localhost:8000", MESSAGES, max_retries=0, **kwargs
        )
    finally:
        render_mod.httpx.post = original_post  # type: ignore[attr-defined]


class TestRenderConversation:
    def test_basic_render(self):
        result = _call_render(_make_transport())
        assert result["token_ids"] == MOCK_TOKEN_IDS
        assert result["loss_mask"] is None  # not yet returned by endpoint

    def test_loss_mask_passthrough(self):
        mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        result = _call_render(_make_transport(assistant_tokens_mask=mask))
        assert result["loss_mask"] == mask

    def test_error_raises(self):
        with pytest.raises(RenderError, match="500"):
            _call_render(_make_transport(status_code=500))

    def test_request_body_forwarding(self):
        transport, body = _capturing_transport()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        _call_render(
            transport,
            tools=tools,
            chat_template_kwargs={"enable_thinking": True},
            max_length=4096,
        )

        assert body["tools"] == tools
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["add_generation_prompt"] is False
        assert body["truncate_prompt_tokens"] == 4096
