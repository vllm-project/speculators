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
