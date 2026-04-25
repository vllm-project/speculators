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


class _DummySyncClient:
    def __init__(self):
        self.completions = _DummySyncCompletions()


class _DummyAsyncClient:
    def __init__(self):
        self.completions = _DummyAsyncCompletions()


def test_generate_hidden_states_text_prompt():
    client = _DummySyncClient()

    result = generate_hidden_states(client, "dummy-model", [1, 2, 3], timeout=1)

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls[0]["prompt"] == [1, 2, 3]


def test_generate_hidden_states_multimodal_prompt():
    client = _DummySyncClient()
    multi_modal_data = {"image": ["https://example.com/cat.png"]}

    result = generate_hidden_states(
        client,
        "dummy-model",
        [4, 5, 6],
        prompt="formatted prompt",
        multi_modal_data=multi_modal_data,
        timeout=1,
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls[0]["prompt"] == {
        "prompt_token_ids": [4, 5, 6],
        "prompt": "formatted prompt",
        "multi_modal_data": multi_modal_data,
    }


def test_generate_hidden_states_async_multimodal_prompt():
    client = _DummyAsyncClient()
    multi_modal_data = {"image": ["https://example.com/cat.png"]}

    result = asyncio.run(
        generate_hidden_states_async(
            client,
            "dummy-model",
            [7, 8, 9],
            prompt="formatted prompt",
            multi_modal_data=multi_modal_data,
            timeout=1,
        )
    )

    assert result == "/tmp/hs_0.safetensors"
    assert client.completions.calls[0]["prompt"] == {
        "prompt_token_ids": [7, 8, 9],
        "prompt": "formatted prompt",
        "multi_modal_data": multi_modal_data,
    }
