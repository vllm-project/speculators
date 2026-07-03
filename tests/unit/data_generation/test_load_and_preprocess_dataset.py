"""Tests for load_and_preprocess_dataset's render-path argument validation.

Both checks fire before target_model_path / train_data_paths are ever touched
(no processor load, no dataset load), so these run with no mocking.
"""

import pytest

from speculators.data_generation.preprocessing import load_and_preprocess_dataset


@pytest.mark.sanity
def test_assistant_pattern_rejected_with_render_endpoint():
    """assistant_pattern is silently unreachable on the render path (it always
    auto-detects its own pattern) -- reject the combination instead of
    dropping the user's pattern without telling them."""
    with pytest.raises(ValueError, match="assistant_pattern"):
        load_and_preprocess_dataset(
            target_model_path="unused",
            train_data_paths=["unused"],
            seq_length=10,
            assistant_pattern="custom-pattern",
            render_endpoint="http://localhost:8000",
        )


@pytest.mark.sanity
def test_chat_template_kwargs_rejected_without_render_endpoint():
    """chat_template_kwargs only reaches build_dataset_from_render; the default
    HF path (build_eagle3_dataset) never receives it. Reject rather than
    silently ignore a value the user explicitly set."""
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        load_and_preprocess_dataset(
            target_model_path="unused",
            train_data_paths=["unused"],
            seq_length=10,
            chat_template_kwargs={"enable_thinking": True},
            render_endpoint=None,
        )
