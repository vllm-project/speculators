"""Configuration registries for data generation pipeline."""

import os
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


NormalizeFn = Callable[[dict], dict]


@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for loading a dataset.

    Datasets already in the canonical ``conversations`` schema (or using the
    OpenAI-style ``messages`` key, which is renamed to ``conversations``
    automatically during ingestion) need no normalizer. ``normalize_fn`` is an
    explicit escape hatch for any other schema (e.g. prompt/answer pairs or
    multi-modal datasets).
    """

    name: str
    hf_path: str
    subset: str | None = None
    split: str
    filter_fn: Callable[[dict], bool] | None = None
    normalize_fn: NormalizeFn | None = None
    # Bare user-prompt column, used when a row has no conversation.
    prompt_field: str | None = None


def _normalize_gsm8k(example: dict) -> dict:
    """Build a conversation from a GSM8K ``question``/``answer`` pair."""
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


def get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _parse_sharegpt4v_part(part: str, image_path: str):
    if part == "<image>":
        return {"type": "image", "path": image_path}

    return {"type": "text", "text": part}


def _parse_sharegpt4v_user_content(content: str, image_path: str):
    return [_parse_sharegpt4v_part(part, image_path) for part in content.split("\n")]


def _parse_sharegpt4v_assistant_content(content: str):
    return [{"type": "text", "text": content}]


def _filter_sharegpt4v_coco(example: dict) -> bool:
    return example["image"].startswith("coco/")


def _normalize_sharegpt4v_coco(example: dict) -> dict:
    coco_dir = get_coco_dir()
    image_path = os.path.join(coco_dir, example["image"].removeprefix("coco/"))

    if not os.path.exists(image_path):
        state_str = "set to" if os.getenv("COCO_DIR") else "default"

        raise ValueError(
            f"No image found at <{image_path}>. "
            f"Please download COCO 2017 Train Images from "
            f"<http://images.cocodataset.org/zips/train2017.zip> and place the "
            f"extracted folder under `COCO_DIR` ({state_str}: `{coco_dir}`)."
        )

    messages = [
        (
            turn
            | {
                "value": (
                    _parse_sharegpt4v_user_content(turn["value"], image_path)
                    if turn["from"] in ("human", "user")
                    else _parse_sharegpt4v_assistant_content(turn["value"])
                )
            }
        )
        for turn in example["conversations"]
    ]

    return {"conversations": messages}


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
        prompt_field="prompt",
        # 'messages' column is renamed to 'conversations' automatically.
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
    # NOTE: You need to serve vLLM with `--allowed-local-media-path /path/to/coco`
    "sharegpt4v_coco": DatasetConfig(
        name="sharegpt4v_coco",
        hf_path="Lin-Chen/ShareGPT4V",
        subset="ShareGPT4V",
        split="train",
        filter_fn=_filter_sharegpt4v_coco,
        normalize_fn=_normalize_sharegpt4v_coco,
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
