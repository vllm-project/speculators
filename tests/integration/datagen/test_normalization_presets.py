"""Network-gated validation that schema normalization yields well-formed
conversations on small slices of the real preset datasets.

Each case streams a few rows from the HuggingFace hub, runs the production
normalization path (the preset normalize_fn, the messages -> conversations
rename, and _normalize_conversation), and asserts the result is a non-empty list
of canonical role/content turns. Tests self-skip when a dataset can't be reached
(offline) or is gated (no token), so they are safe to run anywhere and exercise
real schemas in CI.
"""

import itertools

import pytest
from datasets import Dataset as HFDataset
from datasets import load_dataset

from speculators.data_generation.configs import DatasetConfig, _normalize_gsm8k
from speculators.data_generation.preprocessing import (
    _normalize_conversation,
    _rename_messages_to_conversations,
)

_VALID_ROLES = {"user", "assistant", "system", "tool"}

# Config used to drive normalization for each preset's real schema. nemotron is
# included here for validation only; the shipped preset lives with the discovery
# work (#661/#675), not this PR.
_PRESET_CASES = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        subset="main",
        split="train",
        normalize_fn=_normalize_gsm8k,
    ),
    "nemotron": DatasetConfig(
        name="nemotron",
        hf_path="nvidia/Nemotron-Post-Training-Dataset-v2",
        split="chat",
    ),
}


def _stream_rows(config: DatasetConfig, n: int = 3) -> list[dict]:
    try:
        stream = load_dataset(
            config.hf_path, name=config.subset, split=config.split, streaming=True
        )
        rows = list(itertools.islice(stream, n))
    except Exception as exc:  # noqa: BLE001 - offline / gated / transient hub error
        pytest.skip(f"Could not stream {config.hf_path}:{config.split} ({exc})")
    if not rows:
        pytest.skip(f"No rows streamed from {config.hf_path}:{config.split}")
    return rows


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_PRESET_CASES))
def test_normalization_produces_valid_conversations(name: str):
    config = _PRESET_CASES[name]
    rows = _stream_rows(config)

    dataset = HFDataset.from_list(rows)
    normalize_fn = config.normalize_fn
    if normalize_fn is not None:
        dataset = dataset.map(normalize_fn)
    dataset = _rename_messages_to_conversations(dataset)

    assert "conversations" in dataset.column_names, (
        f"{name}: no 'conversations' column after normalization "
        f"(columns: {dataset.column_names})"
    )

    for conv in dataset["conversations"]:
        normalized = _normalize_conversation(conv)
        assert normalized, f"{name}: normalization produced an empty conversation"
        for turn in normalized:
            role = turn["role"]
            assert role in _VALID_ROLES, f"{name}: invalid role {role!r}"
            assert isinstance(turn["content"], (str, list))
