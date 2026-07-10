import asyncio
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


class _DummyChoice:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class _DummyCompletion:
    def __init__(
        self,
        prompt_token_ids,
        hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.choices = [_DummyChoice(prompt_token_ids)]
        self.kv_transfer_params = {"hidden_states_path": hidden_states_path}


class _DummyChatCompletion:
    def __init__(
        self,
        prompt_token_ids,
        hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.prompt_token_ids = prompt_token_ids
        self.kv_transfer_params = {"hidden_states_path": hidden_states_path}


class _FakeAPIStatusError(Exception):
    """Lightweight stand-in for the SDK's public APIStatusError contract."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeAPIConnectionError(Exception):
    """Lightweight transport error used without constructing SDK internals."""


class _DummySyncCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["prompt"]
        prompt_token_ids = (
            prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt
        )
        return _DummyCompletion(prompt_token_ids)


class _DummyAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["prompt"]
        prompt_token_ids = (
            prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt
        )
        return _DummyCompletion(prompt_token_ids)


class _DummySyncChatCompletions:
    def __init__(
        self,
        prompt_token_ids=None,
        hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.calls = []
        self.prompt_token_ids = prompt_token_ids or [4, 5, 6]
        self.hidden_states_path = hidden_states_path

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _DummyChatCompletion(self.prompt_token_ids, self.hidden_states_path)


class _DummyAsyncChatCompletions:
    def __init__(
        self,
        prompt_token_ids=None,
        hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.calls = []
        self.prompt_token_ids = prompt_token_ids or [7, 8, 9]
        self.hidden_states_path = hidden_states_path

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _DummyChatCompletion(self.prompt_token_ids, self.hidden_states_path)


class _DummySyncChat:
    def __init__(
        self,
        prompt_token_ids=None,
        hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.completions = _DummySyncChatCompletions(
            prompt_token_ids, hidden_states_path
        )


class _DummyAsyncChat:
    def __init__(
        self,
        prompt_token_ids=None,
        hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.completions = _DummyAsyncChatCompletions(
            prompt_token_ids, hidden_states_path
        )


class _DummySyncClient:
    def __init__(
        self,
        chat_prompt_token_ids=None,
        chat_hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.completions = _DummySyncCompletions()
        self.chat = _DummySyncChat(chat_prompt_token_ids, chat_hidden_states_path)


class _DummyAsyncClient:
    def __init__(
        self,
        chat_prompt_token_ids=None,
        chat_hidden_states_path="/tmp/hs_0.safetensors",
    ):
        self.completions = _DummyAsyncCompletions()
        self.chat = _DummyAsyncChat(chat_prompt_token_ids, chat_hidden_states_path)


def test_generate_hidden_states_text_prompt():
    """Text-only items stay on the token-id Completions path."""
    client = _DummySyncClient()

    result = generate_hidden_states(
        client, "dummy-model", {"input_ids": [1, 2, 3]}, timeout=1
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls[0]["prompt"] == [1, 2, 3]


def test_generate_hidden_states_multimodal_messages():
    """Multimodal items are converted to Chat Completions messages."""
    client = _DummySyncClient()
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
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]

    result = generate_hidden_states(
        client,
        "dummy-model",
        {"input_ids": [4, 5, 6], "messages": messages, "tools": tools},
        timeout=1,
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.chat.completions.calls[0]["messages"] == expected_messages
    assert client.chat.completions.calls[0]["extra_body"] == {
        "return_token_ids": True,
        "add_generation_prompt": False,
    }
    assert client.chat.completions.calls[0]["tools"] == tools


def test_generate_hidden_states_multimodal_messages_uses_local_file_url(tmp_path):
    """Local image paths are sent as file URLs for vLLM media loading."""
    client = _DummySyncClient()
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": "A cat."},
    ]

    result = generate_hidden_states(
        client,
        "dummy-model",
        {"input_ids": [4, 5, 6], "messages": messages},
        timeout=1,
    )

    sent_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert result == "/tmp/hs_0.safetensors"
    assert sent_content[0]["type"] == "image_url"
    assert sent_content[0]["image_url"]["url"] == image_path.resolve().as_uri()
    assert sent_content[1] == {"type": "text", "text": "describe"}
    assert "tools" not in client.chat.completions.calls[0]


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
def test_generate_hidden_states_rejects_ambiguous_or_unsupported_image_uri(
    image_ref,
):
    client = _DummySyncClient()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image_ref}],
        }
    ]

    with pytest.raises(ValueError, match="Unsupported|query/fragment|absolute"):
        generate_hidden_states(
            client,
            "dummy-model",
            {"input_ids": [4, 5, 6], "messages": messages},
            timeout=1,
        )

    assert client.chat.completions.calls == []


def test_generate_hidden_states_canonicalizes_percent_encoded_file_uri(tmp_path):
    client = _DummySyncClient()
    image_path = tmp_path / "图 像#100%.png"
    image_path.write_bytes(b"image")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_path.as_uri()},
                }
            ],
        }
    ]

    generate_hidden_states(
        client,
        "dummy-model",
        {"input_ids": [4, 5, 6], "messages": messages},
        timeout=1,
    )

    sent_url = client.chat.completions.calls[0]["messages"][0]["content"][0][
        "image_url"
    ]["url"]
    assert sent_url == image_path.resolve().as_uri()


def test_generate_hidden_states_preserves_extra_chat_message_fields():
    """Tool-call metadata should survive multimodal message conversion."""
    client = _DummySyncClient()
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Calling tool"}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "result"}],
        },
    ]

    generate_hidden_states(
        client,
        "dummy-model",
        {"input_ids": [4, 5, 6], "messages": messages},
        timeout=1,
    )

    sent_messages = client.chat.completions.calls[0]["messages"]
    assert sent_messages[0]["tool_calls"] == messages[0]["tool_calls"]
    assert sent_messages[1]["tool_call_id"] == "call_1"


def test_generate_hidden_states_async_multimodal_messages():
    """Async multimodal generation uses the same Chat Completions payload."""
    client = _DummyAsyncClient()
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
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]

    result = asyncio.run(
        generate_hidden_states_async(
            client,
            "dummy-model",
            {"input_ids": [7, 8, 9], "messages": messages, "tools": tools},
            timeout=1,
        )
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.chat.completions.calls[0]["messages"] == expected_messages
    assert client.chat.completions.calls[0]["extra_body"] == {
        "return_token_ids": True,
        "add_generation_prompt": False,
    }
    assert client.chat.completions.calls[0]["tools"] == tools

    client_without_tools = _DummyAsyncClient()
    asyncio.run(
        generate_hidden_states_async(
            client_without_tools,
            "dummy-model",
            {"input_ids": [7, 8, 9], "messages": messages},
            timeout=1,
        )
    )
    assert "tools" not in client_without_tools.chat.completions.calls[0]


def test_generate_hidden_states_accepts_multimodal_prefix_match_without_rewrite(
    tmp_path,
):
    """Prefix matches are accepted without rewriting locked safetensors files."""
    hs_path = tmp_path / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "hidden_states": torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(
                5, 2, 3
            ),
        },
        hs_path,
    )
    client = _DummySyncClient(
        chat_prompt_token_ids=[1, 2, 3, 4, 5],
        chat_hidden_states_path=str(hs_path),
    )

    result = generate_hidden_states(
        client,
        "dummy-model",
        {
            "input_ids": [1, 2, 3],
            "messages": [{"role": "user", "content": "describe"}],
        },
        timeout=1,
    )

    tensors = load_file(result)
    assert result == str(hs_path)
    assert tensors["token_ids"].tolist() == [1, 2, 3, 4, 5]
    assert tensors["hidden_states"].shape == (5, 2, 3)


def test_generate_hidden_states_rejects_multimodal_non_prefix_mismatch(tmp_path):
    """Non-prefix multimodal token mismatches fail fast."""
    hs_path = tmp_path / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1, 9, 3, 4, 5], dtype=torch.long),
            "hidden_states": torch.zeros(5, 2, 3),
        },
        hs_path,
    )
    client = _DummySyncClient(
        chat_prompt_token_ids=[1, 9, 3, 4, 5],
        chat_hidden_states_path=str(hs_path),
    )

    with pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch"):
        generate_hidden_states(
            client,
            "dummy-model",
            {
                "input_ids": [1, 2, 3],
                "messages": [{"role": "user", "content": "describe"}],
            },
            timeout=1,
        )


def test_generate_hidden_states_text_path_rejects_prefix_mismatch():
    """Text completions require exact token IDs, not prefix matches."""

    class _PrefixTextCompletions:
        calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            return _DummyCompletion([1, 2, 3, 4])

    client: Any = _DummySyncClient()
    client.completions = _PrefixTextCompletions()

    with pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch"):
        generate_hidden_states(
            client,
            "dummy-model",
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=3,
        )

    assert client.completions.calls == 1


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, False),
        (408, True),
        (409, True),
        (425, True),
        (422, False),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_retryable_error_classifies_openai_status_codes(
    monkeypatch,
    status_code,
    expected,
):
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIStatusError",
        _FakeAPIStatusError,
    )

    assert (
        vllm_client_module._is_retryable_error(_FakeAPIStatusError(status_code))
        is expected
    )


def test_retryable_error_accepts_explicit_retry_marker():
    assert vllm_client_module._is_retryable_error(RetryableRequestError("retry"))


def test_generate_hidden_states_does_not_retry_openai_400(monkeypatch):
    class _BadRequestCompletions:
        calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            raise _FakeAPIStatusError(400)

    client: Any = _DummySyncClient()
    client.completions = _BadRequestCompletions()
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIStatusError",
        _FakeAPIStatusError,
    )
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    with pytest.raises(_FakeAPIStatusError, match="HTTP 400"):
        generate_hidden_states(
            client,
            "dummy-model",
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=3,
        )

    assert client.completions.calls == 1


@pytest.mark.parametrize(
    "error_type",
    [
        ValueError,
        TypeError,
        KeyError,
        FileNotFoundError,
        PermissionError,
        RuntimeError,
    ],
)
def test_generate_hidden_states_does_not_retry_unknown_deterministic_error(
    monkeypatch,
    error_type,
):
    class _FailingCompletions:
        calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            raise error_type("deterministic failure")

    client: Any = _DummySyncClient()
    client.completions = _FailingCompletions()
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    with pytest.raises(error_type):
        generate_hidden_states(
            client,
            "dummy-model",
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=3,
        )

    assert client.completions.calls == 1


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_generate_hidden_states_retries_transient_openai_status(
    monkeypatch,
    status_code,
):
    class _FlakyStatusCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _FakeAPIStatusError(status_code)
            return _DummyCompletion(kwargs["prompt"])

    client: Any = _DummySyncClient()
    client.completions = _FlakyStatusCompletions()
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIStatusError",
        _FakeAPIStatusError,
    )
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    result = generate_hidden_states(
        client,
        "dummy-model",
        {"input_ids": [1, 2, 3]},
        timeout=1,
        max_retries=1,
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls == 2


@pytest.mark.parametrize(
    ("status_code", "expected_calls"),
    [(400, 1), (429, 2), (503, 2)],
)
def test_generate_hidden_states_async_matches_status_retry_policy(
    monkeypatch,
    status_code,
    expected_calls,
):
    class _StatusAsyncCompletions:
        calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _FakeAPIStatusError(status_code)
            return _DummyCompletion(kwargs["prompt"])

    client: Any = _DummyAsyncClient()
    client.completions = _StatusAsyncCompletions()
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIStatusError",
        _FakeAPIStatusError,
    )
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    request = generate_hidden_states_async(
        client,
        "dummy-model",
        {"input_ids": [1, 2, 3]},
        timeout=1,
        max_retries=1,
    )
    if status_code == 400:
        with pytest.raises(_FakeAPIStatusError, match="HTTP 400"):
            asyncio.run(request)
    else:
        assert asyncio.run(request) == "/tmp/hs_0.safetensors"

    assert client.completions.calls == expected_calls


@pytest.mark.parametrize("error_type", [TimeoutError, _FakeAPIConnectionError])
def test_generate_hidden_states_consumes_max_retries(monkeypatch, error_type):
    """The retry wrapper consumes max_retries before calling the request helper."""

    class _FlakyCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise error_type("transient transport failure")
            return _DummyCompletion(kwargs["prompt"])

    client: Any = _DummySyncClient()
    client.completions = _FlakyCompletions()
    monkeypatch.setattr(
        vllm_client_module.openai,
        "APIConnectionError",
        _FakeAPIConnectionError,
    )
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    result = generate_hidden_states(
        client,
        "dummy-model",
        {"input_ids": [1, 2, 3]},
        timeout=1,
        max_retries=1,
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls == 2


def test_generate_hidden_states_max_retries_zero_attempts_once(monkeypatch):
    """max_retries=0 performs one request and re-raises its failure."""

    class _FailingCompletions:
        calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            raise TimeoutError("persistent timeout")

    client: Any = _DummySyncClient()
    client.completions = _FailingCompletions()
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    with pytest.raises(TimeoutError, match="persistent timeout"):
        generate_hidden_states(
            client,
            "dummy-model",
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=0,
        )

    assert client.completions.calls == 1


@pytest.mark.parametrize("invalid_max_retries", [-1, True, 1.5, "1"])
def test_generate_hidden_states_rejects_invalid_max_retries_before_request(
    invalid_max_retries,
):
    client = _DummySyncClient()

    with pytest.raises(ValueError, match="non-negative integer"):
        generate_hidden_states(
            client,
            "dummy-model",
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=invalid_max_retries,
        )

    assert client.completions.calls == []


@pytest.mark.parametrize("invalid_max_retries", [-1, True, 1.5, "1"])
def test_generate_hidden_states_async_rejects_invalid_max_retries_before_request(
    invalid_max_retries,
):
    client = _DummyAsyncClient()

    with pytest.raises(ValueError, match="non-negative integer"):
        asyncio.run(
            generate_hidden_states_async(
                client,
                "dummy-model",
                {"input_ids": [1, 2, 3]},
                timeout=1,
                max_retries=invalid_max_retries,
            )
        )

    assert client.completions.calls == []


def test_generate_hidden_states_async_consumes_max_retries(monkeypatch):
    """The async retry wrapper accepts the same max_retries contract."""

    class _FlakyAsyncCompletions:
        calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient timeout")
            return _DummyCompletion(kwargs["prompt"])

    client: Any = _DummyAsyncClient()
    client.completions = _FlakyAsyncCompletions()
    monkeypatch.setattr(vllm_client_module, "RETRY_BACKOFF_BASE", 0)

    result = asyncio.run(
        generate_hidden_states_async(
            client,
            "dummy-model",
            {"input_ids": [1, 2, 3]},
            timeout=1,
            max_retries=1,
        )
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls == 2
