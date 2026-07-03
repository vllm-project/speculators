"""Tests for render_client.py."""

import json

import httpx
import pytest

import speculators.data_generation.render_client as render_mod
from speculators.data_generation.render_client import (
    RenderError,
    render_conversation,
)
from speculators.data_generation.vllm_client import InvalidResponseError

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


def _make_transport(*, token_ids=None, assistant_tokens_mask=None):
    def handler(request: httpx.Request) -> httpx.Response:
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


def _counting_transport(*, status_code=200, body=None):
    """Return (transport, call_counter) -- counter tracks requests received,
    to assert whether a failure was retried."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if status_code != 200:
            return httpx.Response(status_code, text="error from mock")
        return httpx.Response(200, json=body or {"token_ids": MOCK_TOKEN_IDS})

    return httpx.MockTransport(handler), calls


def _call_render(transport, *, max_retries=0, **kwargs):
    original_post = httpx.post

    def mock_post(url, **kw):
        return httpx.Client(transport=transport).request("POST", url, **kw)

    render_mod.httpx.post = mock_post  # type: ignore[attr-defined]
    try:
        return render_conversation(
            "http://localhost:8000", MESSAGES, max_retries=max_retries, **kwargs
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

    @pytest.mark.parametrize(
        ("status_code", "body", "max_retries", "expected_exc", "expected_calls"),
        [
            # 4xx is deterministic (bad request, wrong URL) -- retrying wastes
            # requests without changing the outcome.
            (400, None, 3, InvalidResponseError, 1),
            # 5xx may be transient -- goes through the normal retry policy
            # (initial attempt + max_retries).
            (500, None, 2, RenderError, 3),
            # 200 but missing token_ids must raise RenderError, not a bare
            # KeyError that build_dataset_from_render's except clause can't
            # catch -- and it's not exempt from the retry policy either.
            (200, {"assistant_tokens_mask": [0, 1]}, 1, RenderError, 2),
        ],
    )
    def test_error_handling_and_retry_policy(
        self, status_code, body, max_retries, expected_exc, expected_calls
    ):
        transport, calls = _counting_transport(status_code=status_code, body=body)
        with pytest.raises(expected_exc):
            _call_render(transport, max_retries=max_retries)
        assert calls["count"] == expected_calls

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
