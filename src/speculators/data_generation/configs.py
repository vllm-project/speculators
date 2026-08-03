"""Raw source presets for target-model response regeneration."""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


@dataclass(kw_only=True)
class DatasetConfig:
    """A source dataset whose original assistant turns are never trained on."""

    name: str
    hf_path: str
    subset: str | None = None
    split: str
    filter_fn: Callable[[dict], bool] | None = None
    normalize_fn: Callable[[dict], dict] | None = None
    # Bare user-prompt column, used when a row has no conversation.
    prompt_field: str | None = None


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


def _normalize_gsm8k(example: dict) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def _normalize_nemotron(example: dict) -> dict:
    """Build a conversation from Nemotron ``input`` turns plus ``output``."""
    return {
        "conversations": [
            *example["input"],
            {"role": "assistant", "content": example["output"]},
        ]
    }


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        normalize_fn=_normalize_ultrachat,
        prompt_field="prompt",
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        subset="main",
        split="train",
        normalize_fn=_normalize_gsm8k,
        prompt_field="question",
    ),
    "magpie": DatasetConfig(
        name="magpie",
        hf_path="Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        split="train",
        prompt_field="instruction",
    ),
    "nemotron": DatasetConfig(
        name="nemotron",
        hf_path="nvidia/Llama-Nemotron-Post-Training-Dataset",
        subset="SFT",
        split="chat",
        normalize_fn=_normalize_nemotron,
    ),
    "open-perfectblend": DatasetConfig(
        name="open-perfectblend",
        hf_path="mlabonne/open-perfectblend",
        split="train",
    ),
    # Multi-turn function-calling SFT
    "hermes-fc": DatasetConfig(
        name="hermes-fc",
        hf_path="NousResearch/hermes-function-calling-v1",
        subset="func_calling",
        split="train",
    ),
}
