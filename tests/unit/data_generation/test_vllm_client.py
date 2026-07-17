import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from speculators.data_generation.vllm_client import (
    InvalidResponseError,
    generate_hidden_states,
    generate_hidden_states_async,
)


class _RequestState:
    def __init__(
        self,
        *,
        response_ids: list[int] | None = None,
        nested_prompt_ids: bool = False,
    ):
        self.response_ids = response_ids
        self.nested_prompt_ids = nested_prompt_ids
        self.calls: list[dict[str, Any]] = []

    def request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(kwargs)
        prompt_ids = self.response_ids
        if prompt_ids is None:
            prompt_ids = list(kwargs["prompt"])
        response: dict[str, Any] = {
            "kv_transfer_params": {"hidden_states_path": "/tmp/hs_0.safetensors"}
        }
        if self.nested_prompt_ids:
            response["choices"] = [{"prompt_token_ids": prompt_ids}]
        else:
            response["prompt_token_ids"] = prompt_ids
        return response


class _SyncEndpoint:
    def __init__(self, state: _RequestState):
        self.state = state

    def create(self, **kwargs):
        return self.state.request(kwargs)


class _AsyncEndpoint:
    def __init__(self, state: _RequestState):
        self.state = state

    async def create(self, **kwargs):
        return self.state.request(kwargs)


def _make_client(
    mode: str,
    *,
    text_response_ids: list[int] | None = None,
    chat_response_ids: list[int] | None = None,
) -> tuple[Any, _RequestState, _RequestState]:
    endpoint = _AsyncEndpoint if mode == "async" else _SyncEndpoint
    text_state = _RequestState(
        response_ids=text_response_ids,
        nested_prompt_ids=True,
    )
    chat_state = _RequestState(response_ids=chat_response_ids)
    client = SimpleNamespace(
        completions=endpoint(text_state),
        chat=SimpleNamespace(completions=endpoint(chat_state)),
    )
    return client, text_state, chat_state


def _generate(mode: str, client: Any, item: dict, **kwargs) -> str:
    kwargs.setdefault("max_retries", 0)
    if mode == "async":
        return asyncio.run(
            generate_hidden_states_async(client, "dummy-model", item, **kwargs)
        )
    return generate_hidden_states(client, "dummy-model", item, **kwargs)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_routes_text_and_multimodal_payloads(mode):
    client, text_state, chat_state = _make_client(
        mode,
        chat_response_ids=[4, 5, 6],
    )

    assert (
        _generate(mode, client, {"input_ids": [1, 2, 3]}, timeout=1)
        == "/tmp/hs_0.safetensors"
    )
    assert text_state.calls[0]["prompt"] == [1, 2, 3]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "describe_image",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://example.com/cat.png"},
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": "A cat."},
    ]

    assert (
        _generate(
            mode,
            client,
            {"input_ids": [4, 5, 6], "messages": messages, "tools": tools},
            timeout=1,
        )
        == "/tmp/hs_0.safetensors"
    )
    chat_call = chat_state.calls[0]
    assert chat_call["messages"][0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        },
        {"type": "text", "text": "describe"},
    ]
    assert chat_call["tools"] == tools
    assert chat_call["extra_body"] == {
        "add_generation_prompt": False,
        "continue_final_message": True,
        "return_token_ids": True,
    }

    _generate(mode, client, {"input_ids": [4, 5, 6], "messages": messages})
    assert "tools" not in chat_state.calls[1]


def test_canonicalizes_local_paths_and_percent_encoded_file_uris(tmp_path):
    client, _, chat_state = _make_client("sync", chat_response_ids=[1])
    image_path = tmp_path / "test image#100%.png"
    image_path.write_bytes(b"image")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
            ],
        }
    ]

    _generate("sync", client, {"input_ids": [1], "messages": messages}, timeout=1)

    sent_content = chat_state.calls[0]["messages"][0]["content"]
    assert [part["image_url"]["url"] for part in sent_content] == [
        image_path.resolve().as_uri(),
        image_path.resolve().as_uri(),
    ]


@pytest.mark.parametrize(
    "image_ref",
    [
        "file://remote.example/tmp/cat.png",
        "file:///tmp/cat.png?version=1",
        "file:relative/cat.png",
        "ftp://example.com/cat.png",
        "//remote.example/tmp/cat.png",
    ],
)
def test_rejects_ambiguous_or_unsupported_image_uri(image_ref):
    client, _, chat_state = _make_client("sync", chat_response_ids=[1])
    item = {
        "input_ids": [1],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_ref}],
            }
        ],
    }

    with pytest.raises(ValueError, match="Unsupported|query/fragment|absolute"):
        _generate("sync", client, item, timeout=1)
    assert chat_state.calls == []


@pytest.mark.parametrize(
    ("mode", "multimodal", "response_ids", "accepted"),
    [
        ("sync", True, [1, 2, 3, 4], True),
        ("async", True, [1, 2, 3, 4], True),
        ("sync", True, [1, 9, 3, 4], True),
        ("sync", False, [1, 2, 3, 4], False),
    ],
)
def test_token_alignment_policy(mode, multimodal, response_ids, accepted):
    client, text_state, chat_state = _make_client(
        mode,
        text_response_ids=response_ids,
        chat_response_ids=response_ids,
    )
    item: dict[str, object] = {"input_ids": [1, 2, 3]}
    if multimodal:
        item["messages"] = [{"role": "user", "content": "describe"}]

    if accepted:
        assert _generate(mode, client, item, timeout=1) == "/tmp/hs_0.safetensors"
    else:
        with pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch"):
            _generate(mode, client, item, timeout=1)

    state = chat_state if multimodal else text_state
    assert len(state.calls) == 1
