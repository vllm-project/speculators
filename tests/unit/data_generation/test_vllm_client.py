import asyncio

from speculators.data_generation.vllm_client import (
    generate_hidden_states,
    generate_hidden_states_async,
)


class _DummyChoice:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class _DummyCompletion:
    def __init__(self, prompt_token_ids):
        self.choices = [_DummyChoice(prompt_token_ids)]
        self.kv_transfer_params = {"hidden_states_path": "/tmp/hs_0.safetensors"}


class _DummyChatCompletion:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids
        self.kv_transfer_params = {"hidden_states_path": "/tmp/hs_0.safetensors"}


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
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _DummyChatCompletion([4, 5, 6])


class _DummyAsyncChatCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _DummyChatCompletion([7, 8, 9])


class _DummySyncChat:
    def __init__(self):
        self.completions = _DummySyncChatCompletions()


class _DummyAsyncChat:
    def __init__(self):
        self.completions = _DummyAsyncChatCompletions()


class _DummySyncClient:
    def __init__(self):
        self.completions = _DummySyncCompletions()
        self.chat = _DummySyncChat()


class _DummyAsyncClient:
    def __init__(self):
        self.completions = _DummyAsyncCompletions()
        self.chat = _DummyAsyncChat()


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
