from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from speculators.data_generation import render_client
from speculators.data_generation.vllm_client import InvalidResponseError


def _resp(status, text="", payload=None):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.return_value = payload or {}
    return r


def test_render_reports_context_limit_without_retry():
    bad = _resp(
        HTTPStatus.BAD_REQUEST,
        text="max_tokens must be at least 1, got 0",
    )

    with (
        patch.object(render_client, "_post", return_value=bad) as post,
        pytest.raises(InvalidResponseError) as exc_info,
    ):
        render_client.render_conversation(
            "http://x",
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
            truncate_prompt_tokens=8192,
        )

    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["truncate_prompt_tokens"] == 8192
    message = str(exc_info.value)
    assert "--max-model-len" in message
    assert "--seq-length" in message
    assert "8192" in message
    assert "max_tokens must be at least 1" in message


def test_reasoning_effort_is_sent_as_top_level_field():
    """The effort rides as a top-level request field, not template kwargs."""

    ok = _resp(HTTPStatus.OK, payload={"token_ids": [1, 2, 3]})

    with patch.object(render_client, "_post", return_value=ok) as post:
        ids = render_client.render_conversation(
            "http://x",
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            reasoning_effort="high",
        )

    assert ids == [1, 2, 3]
    body = post.call_args.kwargs["json"]
    assert body["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in body


def test_reasoning_effort_omitted_when_not_given():
    """No effort given means the request body has no such field."""

    ok = _resp(HTTPStatus.OK, payload={"token_ids": [1]})

    with patch.object(render_client, "_post", return_value=ok) as post:
        render_client.render_conversation(
            "http://x",
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )

    assert "reasoning_effort" not in post.call_args.kwargs["json"]
