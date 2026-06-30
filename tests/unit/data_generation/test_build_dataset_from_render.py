"""Tests for build_dataset_from_render: server-provided and regex-fallback masks."""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    build_dataset_from_render,
    load_processor,
)
from speculators.data_generation.render_client import RenderError

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

CONV = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


@pytest.fixture(scope="module")
def processor():
    return load_processor(MODEL)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def _make_dataset():
    return HFDataset.from_dict({"conversations": [CONV]})


def _mock_render(token_ids, loss_mask=None):
    """Patch render_conversation to return fixed token_ids + loss_mask."""

    def _fake(endpoint, messages, **kwargs):
        return {"token_ids": token_ids, "loss_mask": loss_mask}

    return patch(
        "speculators.data_generation.preprocessing.render_conversation",
        side_effect=_fake,
    )


@pytest.mark.sanity
def test_server_provided_mask_flows_through(processor, tokenizer):
    """When the render endpoint returns a real loss_mask, use it directly."""
    ids = tokenizer.apply_chat_template(
        CONV, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    mask = [0] * len(ids)
    # Mark the last 5 tokens as trainable (simulates assistant span).
    for i in range(len(ids) - 5, len(ids)):
        mask[i] = 1

    with _mock_render(list(ids), loss_mask=mask):
        ds = build_dataset_from_render(
            _make_dataset(), "http://fake", processor, max_length=4096
        )

    assert len(ds) == 1
    assert ds[0]["input_ids"].tolist() == list(ids)
    assert ds[0]["loss_mask"].tolist() == mask
    assert ds[0]["loss_mask"].sum().item() == 5


@pytest.mark.sanity
def test_regex_fallback_when_loss_mask_is_none(processor, tokenizer):
    """When loss_mask=None, fall back to regex-based assistant detection."""
    ids = tokenizer.apply_chat_template(
        CONV, tokenize=True, add_generation_prompt=False, return_dict=False
    )

    with _mock_render(list(ids), loss_mask=None):
        ds = build_dataset_from_render(
            _make_dataset(), "http://fake", processor, max_length=4096
        )

    assert len(ds) == 1
    assert ds[0]["input_ids"].tolist() == list(ids)
    mask = ds[0]["loss_mask"]
    # The regex fallback should mark assistant tokens as trainable.
    assert mask.sum().item() > 0, "regex fallback produced all-zero mask"
    # Verify the masked-in tokens decode to the assistant content.
    masked_ids = [t for t, m in zip(ids, mask.tolist(), strict=False) if m]
    decoded = tokenizer.decode(masked_ids)
    assert "Paris" in decoded


@pytest.mark.sanity
def test_full_server_mask_skips_local_pattern_detection(processor, tokenizer):
    """When the server masks every conversation, the local regex pattern is
    never detected -- so a model whose detection would raise still succeeds."""
    ids = tokenizer.apply_chat_template(
        CONV, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    mask = [0] * len(ids)
    mask[-1] = 1

    with (
        _mock_render(list(ids), loss_mask=mask),
        patch(
            "speculators.data_generation.preprocessing._detect_assistant_pattern",
            side_effect=ValueError("cannot detect"),
        ) as detect,
    ):
        ds = build_dataset_from_render(
            _make_dataset(), "http://fake", processor, max_length=4096
        )

    detect.assert_not_called()
    assert len(ds) == 1
    assert ds[0]["loss_mask"].sum().item() == 1


@pytest.mark.sanity
def test_over_length_render_dropped(processor):
    """vLLM does not truncate multimodal prompts, so the server can return
    token_ids longer than max_length. Such a row is dropped (parity with the
    default path), never emitted as an over-length / misaligned sample."""
    long_ids = list(range(100))  # 100 > max_length (50) below
    with _mock_render(long_ids, loss_mask=None):
        ds = build_dataset_from_render(
            _make_dataset(), "http://fake", processor, max_length=50
        )
    assert len(ds) == 0


@pytest.mark.sanity
def test_render_error_does_not_crash_the_run(processor):
    """A per-conversation render failure is skipped, not propagated -- one bad
    row must not abort the whole dataset build (parity with the default path)."""
    with patch(
        "speculators.data_generation.preprocessing.render_conversation",
        side_effect=RenderError("boom"),
    ):
        ds = build_dataset_from_render(
            _make_dataset(), "http://fake", processor, max_length=4096
        )
    assert len(ds) == 0


@pytest.fixture(scope="module")
def mm_processor():
    return load_processor("Qwen/Qwen3-VL-2B-Instruct")


@pytest.mark.sanity
def test_multimodal_keeps_messages_for_extraction(mm_processor):
    """Multimodal (ProcessorMixin) rows keep a `messages` column so hidden-state
    extraction can re-send the conversation with images via the chat API --
    token_ids alone cannot carry the image."""
    mm_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "path": "/tmp/x.png"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]
    ids = list(range(12))
    mask = [0] * 9 + [1, 1, 1]
    with _mock_render(ids, loss_mask=mask):
        ds = build_dataset_from_render(
            HFDataset.from_dict({"conversations": [mm_conv]}),
            "http://fake",
            mm_processor,
            max_length=4096,
        )
    assert "messages" in ds.column_names
    assert len(ds) == 1
    assert ds[0]["messages"][0]["role"] == "user"


@pytest.mark.sanity
def test_text_model_has_no_messages_column(processor):
    """Text-only processors must not add a `messages` column (parity with the
    default path, which only adds it for ProcessorMixin)."""
    with _mock_render(list(range(10)), loss_mask=[0] * 7 + [1, 1, 1]):
        ds = build_dataset_from_render(
            _make_dataset(), "http://fake", processor, max_length=4096
        )
    assert "messages" not in ds.column_names
