"""All data producers share validation and finalization at dataset construction."""

import logging
from typing import Any

import pytest
import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk

from speculators.cli.regenerate_responses import _sample_from_response
from speculators.data_generation import preprocessing
from speculators.data_generation.preprocessing import build_speculator_training_dataset
from speculators.train.data import build_client_item


@pytest.mark.parametrize("source", ["prepared", "regenerated", "rendered"])
@pytest.mark.parametrize(
    ("max_length", "minimum_valid_tokens", "expected_ids", "expected_mask"),
    [
        (3, 1, [10, 11, 20], [0, 0, 1]),
        (2, 0, [], []),  # truncation removes all supervision, even with minimum=0
        (3, 2, [], []),  # remaining supervision is below the minimum
    ],
)
def test_all_sources_share_truncation_and_filtering(
    monkeypatch, source, max_length, minimum_valid_tokens, expected_ids, expected_mask
):
    ids, mask = [10, 11, 20, 21], [0, 0, 1, 1]

    def render(endpoint, messages, *, add_generation_prompt, **kwargs):
        assert source == "rendered", "Prepared records must not be re-rendered"
        tokens = ids[:2] if add_generation_prompt else ids
        return tokens[: kwargs["truncate_prompt_tokens"]]

    monkeypatch.setattr(preprocessing, "render_conversation", render)
    row: dict[str, Any]
    if source == "rendered":
        row = {
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    elif source == "regenerated":
        row, _, _ = _sample_from_response(
            {
                "prompt_token_ids": ids[:2],
                "choices": [{"token_ids": ids[2:], "message": {"content": "answer"}}],
            },
            conv_id="c",
            sample_index=0,
            idx=0,
            endpoint="ep",
            sampling_params={},
        )
    else:
        row = {"input_ids": ids, "loss_mask": mask}

    dataset = build_speculator_training_dataset(
        HFDataset.from_list([row]),
        num_proc=1,
        max_length=max_length,
        minimum_valid_tokens=minimum_valid_tokens,
        render_endpoint="http://render" if source == "rendered" else None,
    )
    assert len(dataset) == bool(expected_ids)
    if expected_ids:
        sample = dataset[0]
        assert sample["input_ids"].tolist() == expected_ids
        assert sample["loss_mask"].tolist() == expected_mask
        assert sample["input_ids"].dtype == sample["loss_mask"].dtype == torch.long
        assert sample["seq_len"] == len(expected_ids)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        ([0, 1], "shape mismatch"),
        ([0, 1, 2], "only 0 and 1"),
        ([0, 1, -1], "only 0 and 1"),
        ([0, 1, 0.5], "only 0 and 1"),
    ],
)
def test_prepared_rows_validate_before_truncation(mask, message):
    dataset = HFDataset.from_dict({"input_ids": [[10, 20, 21]], "loss_mask": [mask]})
    with pytest.raises(ValueError, match=message):
        build_speculator_training_dataset(dataset, max_length=2, num_proc=1)


@pytest.mark.parametrize(
    ("rendered", "length", "warns"),
    [(False, 4, False), (False, 5, True), (True, 4, True), (True, 3, False)],
)
def test_truncation_warning_distinguishes_server_limit(caplog, rendered, length, warns):
    with caplog.at_level(logging.WARNING):
        result = preprocessing._finalize_samples(
            [{"input_ids": list(range(length)), "loss_mask": [0] + [1] * (length - 1)}],
            max_length=4,
            rendered=rendered,
        )
    assert bool(result["input_ids"])
    assert ("may have been truncated" in caplog.text) == warns


def test_filtering_keeps_media_attached_to_the_correct_row(tmp_path):
    def messages(image):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"https://x/{image}"}}
                ],
            }
        ]

    dataset = HFDataset.from_list(
        [
            {
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 0, 1],
                "messages": messages("drop"),
            },
            {
                "input_ids": [4, 5, 6],
                "loss_mask": [0, 1, 1],
                "messages": messages("keep"),
            },
            {"input_ids": [7, 8, 9], "loss_mask": [0, 1, 1], "messages": None},
        ]
    )
    prepared = build_speculator_training_dataset(
        dataset, num_proc=1, max_length=3, minimum_valid_tokens=2
    )
    prepared.save_to_disk(str(tmp_path / "prepared"))
    saved = load_from_disk(str(tmp_path / "prepared"))
    assert len(saved) == 2
    assert saved[0]["input_ids"].tolist() == [4, 5, 6]
    assert saved[0]["messages"] == messages("keep")
    assert build_client_item(saved[0])["messages"] == messages("keep")
    assert build_client_item(saved[1]) == {"input_ids": [7, 8, 9]}


def test_prepared_media_schema_survives_text_only_first_batch():
    # datasets.map batches 1,000 rows. Optional messages must work even when the
    # first batch contains only text and the image first appears in a later one.
    media = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/image"}}],
        }
    ]
    dataset = HFDataset.from_dict(
        {
            "input_ids": [[1, 2]] * 1001,
            "loss_mask": [[0, 1]] * 1001,
            "messages": [None] * 1000 + [media],
        }
    )
    prepared = build_speculator_training_dataset(dataset, num_proc=1)
    assert build_client_item(prepared[0]) == {"input_ids": [1, 2]}
    assert build_client_item(prepared[-1])["messages"] == media
