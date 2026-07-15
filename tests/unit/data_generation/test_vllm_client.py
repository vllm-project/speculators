import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from speculators.data_generation import vllm_client as vllm_client_module
from speculators.data_generation.vllm_client import (
    InvalidResponseError,
    RetryableRequestError,
    generate_hidden_states,
    generate_hidden_states_async,
)


class _FakeAPIStatusError(Exception):
    """Lightweight stand-in for the SDK's public APIStatusError contract."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    """Lightweight transport error used without constructing SDK internals."""


class _RequestState:
    def __init__(
        self,
        *,
        response_ids: list[int] | None = None,
        failures: tuple[Exception, ...] = (),
        nested_prompt_ids: bool = False,
        object_response: bool = False,
        hidden_states_path: str = "/tmp/hs_0.safetensors",
    ):
        self.response_ids = response_ids
        self.failures = list(failures)
        self.nested_prompt_ids = nested_prompt_ids
        self.object_response = object_response
        self.hidden_states_path = hidden_states_path
        self.calls: list[dict[str, Any]] = []

    def request(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)

        prompt_ids = self.response_ids
        if prompt_ids is None:
            prompt_ids = list(kwargs["prompt"])
        response: dict[str, Any] = {
            "kv_transfer_params": {
                "hidden_states_path": self.hidden_states_path,
            }
        }
        if self.nested_prompt_ids:
            response["choices"] = [{"prompt_token_ids": prompt_ids}]
        else:
            response["prompt_token_ids"] = prompt_ids
        if self.object_response:
            response["kv_transfer_params"] = SimpleNamespace(
                **response["kv_transfer_params"]
            )
            if self.nested_prompt_ids:
                response["choices"] = [SimpleNamespace(**response["choices"][0])]
            return SimpleNamespace(**response)
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
    text_failures: tuple[Exception, ...] = (),
    object_response: bool = False,
    hidden_states_path: str = "/tmp/hs_0.safetensors",
) -> tuple[Any, _RequestState, _RequestState]:
    endpoint = _AsyncEndpoint if mode == "async" else _SyncEndpoint
    text_state = _RequestState(
        response_ids=text_response_ids,
        failures=text_failures,
        nested_prompt_ids=True,
        object_response=object_response,
        hidden_states_path=hidden_states_path,
    )
    chat_state = _RequestState(
        response_ids=chat_response_ids,
        object_response=object_response,
        hidden_states_path=hidden_states_path,
    )
    client = SimpleNamespace(
        completions=endpoint(text_state),
        chat=SimpleNamespace(completions=endpoint(chat_state)),
    )
    return client, text_state, chat_state


def _generate(mode: str, client: Any, item: dict, **kwargs) -> str:
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
    text_call = text_state.calls[0]
    assert text_call["prompt"] == [1, 2, 3]
    assert text_call["extra_body"] == {"return_token_ids": True}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "describe_image",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "describe_image", "arguments": "{}"},
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
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "content": "A cat.", "tool_call_id": "call_1"},
    ]
    item = {
        "input_ids": [4, 5, 6],
        "messages": messages,
        "tools": tools,
    }

    assert _generate(mode, client, item, timeout=1) == "/tmp/hs_0.safetensors"
    chat_call = chat_state.calls[0]
    assert chat_call["messages"][0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        },
        {"type": "text", "text": "describe"},
    ]
    assert chat_call["messages"][1]["tool_calls"] == tool_calls
    assert chat_call["messages"][2]["tool_call_id"] == "call_1"
    assert chat_call["tools"] == tools
    assert chat_call["extra_body"] == {
        "add_generation_prompt": False,
        "return_token_ids": True,
    }

    _generate(mode, client, {"input_ids": [4, 5, 6], "messages": messages})
    assert "tools" not in chat_state.calls[1]


def test_canonicalizes_local_paths_and_percent_encoded_file_uris(tmp_path):
    client, _, chat_state = _make_client("sync", chat_response_ids=[1])
    image_path = tmp_path / "图 像#100%.png"
    image_path.write_bytes(b"image")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {
                    "type": "image_url",
                    "image_url": {"url": image_path.as_uri()},
                },
            ],
        }
    ]

    _generate(
        "sync",
        client,
        {"input_ids": [1], "messages": messages},
        timeout=1,
    )

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
        "file:///tmp/cat.png#fragment",
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
        ("sync", True, [1, 9, 3, 4], False),
        ("sync", False, [1, 2, 3, 4], False),
    ],
)
def test_token_alignment_policy(mode, multimodal, response_ids, accepted):
    client, text_state, chat_state = _make_client(
        mode,
        text_response_ids=response_ids,
        chat_response_ids=response_ids,
    )
    item = {"input_ids": [1, 2, 3]}
    if multimodal:
        item["messages"] = [{"role": "user", "content": "describe"}]

    if accepted:
        assert _generate(mode, client, item, timeout=1) == "/tmp/hs_0.safetensors"
    else:
        with pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch"):
            _generate(mode, client, item, timeout=1, max_retries=3)

    state = chat_state if multimodal else text_state
    assert len(state.calls) == 1


def test_object_response_accepts_multimodal_prefix_without_rewriting_file(tmp_path):
    hidden_states_path = tmp_path / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1, 2, 3, 4, 5]),
            "hidden_states": torch.arange(30, dtype=torch.float32).reshape(5, 2, 3),
        },
        hidden_states_path,
    )
    client, _, _ = _make_client(
        "sync",
        chat_response_ids=[1, 2, 3, 4, 5],
        object_response=True,
        hidden_states_path=str(hidden_states_path),
    )

    result = _generate(
        "sync",
        client,
        {
            "input_ids": [1, 2, 3],
            "messages": [{"role": "user", "content": "describe"}],
        },
        timeout=1,
    )

    tensors = load_file(result)
    assert result == str(hidden_states_path)
    assert tensors["token_ids"].tolist() == [1, 2, 3, 4, 5]
    assert tensors["hidden_states"].shape == (5, 2, 3)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_FakeAPIStatusError(400), False),
        (_FakeAPIStatusError(408), True),
        (_FakeAPIStatusError(409), True),
        (_FakeAPIStatusError(425), True),
        (_FakeAPIStatusError(422), False),
        (_FakeAPIStatusError(429), True),
        (_FakeAPIStatusError(500), True),
        (_FakeAPIStatusError(503), True),
        (_FakeAPIConnectionError("connection"), True),
        (TimeoutError("timeout"), True),
        (ConnectionError("connection"), True),
        (RetryableRequestError("retry"), True),
        (InvalidResponseError("invalid"), False),
        (ValueError("deterministic"), False),
    ],
)
def test_retryable_error_classification(monkeypatch, error, expected):
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIStatusError",
        _FakeAPIStatusError,
    )
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIConnectionError",
        _FakeAPIConnectionError,
    )

    assert vllm_client_module._is_retryable_error(error) is expected


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("error_kind", "should_retry", "max_retries"),
    [
        ("status-400", False, 3),
        ("status-503", True, 1),
        ("timeout", True, 1),
        ("value", False, 3),
    ],
)
def test_sync_and_async_retry_policy(
    monkeypatch,
    mode,
    error_kind,
    should_retry,
    max_retries,
):
    errors = {
        "status-400": _FakeAPIStatusError(400),
        "status-503": _FakeAPIStatusError(503),
        "timeout": TimeoutError("transient timeout"),
        "value": ValueError("deterministic failure"),
    }
    error = errors[error_kind]
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIStatusError",
        _FakeAPIStatusError,
    )
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)
    client, text_state, _ = _make_client(
        mode,
        text_failures=(error,),
    )

    if should_retry:
        assert (
            _generate(
                mode,
                client,
                {"input_ids": [1, 2, 3]},
                timeout=1,
                max_retries=max_retries,
            )
            == "/tmp/hs_0.safetensors"
        )
        assert len(text_state.calls) == 2
    else:
        with pytest.raises(type(error)):
            _generate(
                mode,
                client,
                {"input_ids": [1, 2, 3]},
                timeout=1,
                max_retries=max_retries,
            )
        assert len(text_state.calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_max_retries_zero_attempts_once(monkeypatch, mode):
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)
    client, text_state, _ = _make_client(
        mode,
        text_failures=(TimeoutError("persistent timeout"),),
    )

    with pytest.raises(TimeoutError, match="persistent timeout"):
        _generate(
            mode,
            client,
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=0,
        )

    assert len(text_state.calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("invalid_max_retries", [-1, True, 1.5, "1"])
def test_rejects_invalid_max_retries_before_request(mode, invalid_max_retries):
    client, text_state, _ = _make_client(mode)

    with pytest.raises(ValueError, match="non-negative integer"):
        _generate(
            mode,
            client,
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=invalid_max_retries,
        )

    assert text_state.calls == []
