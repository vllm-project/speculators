import openai


class InvalidResponseError(Exception):
    pass


def extract_output(completion, token_ids) -> str:
    prompt_token_ids = getattr(completion.choices[0], "prompt_token_ids", None)

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    if prompt_token_ids != token_ids:
        raise InvalidResponseError(
            f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
        )

    if not hasattr(completion, "kv_transfer_params"):
        raise InvalidResponseError("Response missing kv_transfer_params")

    return completion.kv_transfer_params.get("hidden_states_path")


async def generate_hidden_states_async(
    client: openai.AsyncClient, model: str, token_ids: list[int]
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.
    """

    completion = await client.completions.create(
        model=model,
        prompt=token_ids,
        max_tokens=1,
        extra_body={"return_token_ids": True},
    )

    return extract_output(completion, token_ids)


def generate_hidden_states(
    client: openai.Client, model: str, token_ids: list[int]
) -> str:
    completion = client.completions.create(
        model=model,
        prompt=token_ids,
        max_tokens=1,
        extra_body={"return_token_ids": True},
    )
    return extract_output(completion, token_ids)
