"""Unit tests for conversation preprocessing (server render requests)."""

import pytest

from speculators.data_generation import preprocessing

CONV = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "again"},
    {"role": "assistant", "content": "there"},
]


def _capture_render(calls):
    def render(endpoint, messages, *, add_generation_prompt, **kwargs):
        calls.append((add_generation_prompt, kwargs.get("reasoning_effort")))
        # Render depends only on the message count, so the generation-prompt
        # render of conv[:j] is a prefix of the full render of conv[:j+1].
        return [100 + i for i in range(len(messages))]

    return render


def _run(monkeypatch, examples):
    calls = []
    monkeypatch.setattr(preprocessing, "render_conversation", _capture_render(calls))
    result = preprocessing._preprocess_batch(
        examples,
        is_multimodal=False,
        render_endpoint="http://render.test",
        max_length=64,
    )
    return result, calls


def test_server_render_replays_row_reasoning_effort(monkeypatch):
    """Rows record their generation-time effort in metadata.sampling_params;
    every boundary render must replay it so effort-in-prompt templates
    render the token stream the target model actually saw."""
    result, calls = _run(
        monkeypatch,
        {
            "conversations": [CONV],
            "metadata": [{"sampling_params": {"reasoning_effort": "high"}}],
        },
    )

    assert len(result["input_ids"]) == 2  # one row per assistant turn
    assert calls
    assert all(effort == "high" for _, effort in calls)


def test_server_render_omits_effort_without_metadata(monkeypatch):
    """Rows without metadata render with no reasoning_effort field."""

    result, calls = _run(monkeypatch, {"conversations": [CONV]})

    assert len(result["input_ids"]) == 2
    assert calls
    assert all(effort is None for _, effort in calls)


@pytest.mark.parametrize(
    "bad_metadata",
    [
        "not a dict",
        ["not", "a", "dict"],
        {"sampling_params": "bad"},
        {},
        {"sampling_params": {}},
        {"sampling_params": {"reasoning_effort": ""}},
        {"sampling_params": {"reasoning_effort": 7}},
        None,
    ],
)
def test_server_render_ignores_malformed_metadata(monkeypatch, bad_metadata):
    """Each unusable metadata value degrades to no effort on its own row."""

    result, calls = _run(
        monkeypatch,
        {"conversations": [CONV], "metadata": [bad_metadata]},
    )

    assert len(result["input_ids"]) == 2  # one row per assistant turn
    assert all(effort is None for _, effort in calls)
