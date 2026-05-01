import asyncio

import pytest
import torch
from safetensors.torch import load_file, save_file

from speculators.data_generation.vllm_client import (
    InvalidResponseError,
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
    client = _DummySyncClient()

    result = generate_hidden_states(client, "dummy-model", [1, 2, 3], timeout=1)

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls[0]["prompt"] == [1, 2, 3]


def test_generate_hidden_states_multimodal_messages():
    client = _DummySyncClient()
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
        [4, 5, 6],
        messages=messages,
        timeout=1,
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.chat.completions.calls[0]["messages"] == expected_messages
    assert client.chat.completions.calls[0]["extra_body"] == {
        "return_token_ids": True,
        "add_generation_prompt": False,
    }


def test_generate_hidden_states_multimodal_messages_inlines_local_image(tmp_path):
    client = _DummySyncClient()
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": "A cat."},
    ]

    result = generate_hidden_states(
        client,
        "dummy-model",
        [4, 5, 6],
        messages=messages,
        timeout=1,
    )

    sent_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert result == "/tmp/hs_0.safetensors"
    assert sent_content[0]["type"] == "image_url"
    assert sent_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent_content[1] == {"type": "text", "text": "describe"}


def test_generate_hidden_states_async_multimodal_messages():
    client = _DummyAsyncClient()
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
            [7, 8, 9],
            messages=messages,
            timeout=1,
        )
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.chat.completions.calls[0]["messages"] == expected_messages
    assert client.chat.completions.calls[0]["extra_body"] == {
        "return_token_ids": True,
        "add_generation_prompt": False,
    }


def test_generate_hidden_states_truncates_multimodal_prefix_match(tmp_path):
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
        [1, 2, 3],
        messages=[{"role": "user", "content": "describe"}],
        timeout=1,
    )

    tensors = load_file(result)
    assert result == str(hs_path)
    assert tensors["token_ids"].tolist() == [1, 2, 3]
    assert tensors["hidden_states"].shape == (3, 2, 3)
    assert torch.equal(
        tensors["hidden_states"],
        torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)[:3],
    )


def test_generate_hidden_states_rejects_multimodal_non_prefix_mismatch(tmp_path):
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
            [1, 2, 3],
            messages=[{"role": "user", "content": "describe"}],
            timeout=1,
        )


def test_generate_hidden_states_text_path_rejects_prefix_mismatch():
    class _PrefixTextCompletions:
        def create(self, **kwargs):
            return _DummyCompletion([1, 2, 3, 4])

    client = _DummySyncClient()
    client.completions = _PrefixTextCompletions()

    with pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch"):
        generate_hidden_states(client, "dummy-model", [1, 2, 3], timeout=1)
