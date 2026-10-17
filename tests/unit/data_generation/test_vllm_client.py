"""Hidden-state clients must use and validate the prepared media token window."""

from contextlib import nullcontext
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from speculators.data_generation.vllm_client import (
    ClientItem,
    InvalidResponseError,
    generate_hidden_states,
    generate_hidden_states_async,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mismatched", [False, True])
async def test_audio_hidden_state_prompt_alignment(asynchronous, mismatched):
    token_ids = [1, 2, 2, 3]
    item = cast(
        "ClientItem",
        {
            "input_ids": token_ids,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "file:///audio.wav"}}
                    ],
                },
                {"role": "assistant", "content": "a long transcript"},
            ],
        },
    )
    transfer_params = {"hidden_states_path": "/states/test.safetensors"}
    response = Mock(
        prompt_token_ids=[*token_ids, 99] if mismatched else token_ids,
        kv_transfer_params=transfer_params,
    )
    client = Mock()
    create = (
        AsyncMock(return_value=response)
        if asynchronous
        else Mock(return_value=response)
    )
    client.chat.completions.create = create
    expected = (
        pytest.raises(InvalidResponseError, match="Prompt token IDs mismatch")
        if mismatched
        else nullcontext()
    )
    with expected:
        result = (
            await generate_hidden_states_async(client, "verifier", item, max_retries=0)
            if asynchronous
            else generate_hidden_states(client, "verifier", item, max_retries=0)
        )
        assert result == (transfer_params["hidden_states_path"], transfer_params)
    create.assert_called_once_with(
        model="verifier",
        messages=item["messages"],
        max_tokens=1,
        extra_body={
            "add_generation_prompt": False,
            "continue_final_message": False,
            "truncate_prompt_tokens": len(token_ids),
            "truncation_side": "right",
            "return_token_ids": True,
        },
        timeout=120,
    )
