from http import HTTPStatus
from unittest.mock import MagicMock, patch

from speculators.data_generation import render_client


def _resp(status, text="", payload=None):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.return_value = payload or {}
    return r


def test_render_retries_once_on_max_tokens_zero():
    bad = _resp(
        HTTPStatus.BAD_REQUEST,
        text="max_tokens must be at least 1, got 0",
    )
    ok = _resp(
        HTTPStatus.OK,
        payload={"token_ids": [1, 2, 3]},
    )

    with patch.object(render_client, "_post", side_effect=[bad, ok]) as post:
        out = render_client.render_conversation(
            "http://x",
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
            truncate_prompt_tokens=8192,
        )

    assert out == [1, 2, 3]
    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"]["truncate_prompt_tokens"] == 8191
