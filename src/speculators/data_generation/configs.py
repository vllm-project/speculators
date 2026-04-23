"""Configuration registries for data generation pipeline."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset"""

    name: str
    hf_path: str
    split: str
    normalize_fn: Callable[[dict], dict] | None = None


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


def _normalize_media_ref(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    filename = getattr(value, "filename", None)
    if filename:
        return str(filename)
    return value


def _split_llava_user_content(text: str, image_ref: Any) -> list[dict[str, Any]]:
    parts = re.split(r"(<image>)", text or "")
    content: list[dict[str, Any]] = []
    inserted_image = False

    for part in parts:
        if part == "<image>":
            content.append({"type": "image", "image": image_ref})
            inserted_image = True
        elif part:
            # Preserve surrounding whitespace / newlines (e.g. "<image>\n..."),
            # which some chat templates rely on as turn-level separators.
            # Only use ``.strip()`` as the filter predicate below.
            content.append({"type": "text", "text": part})

    if not inserted_image and image_ref is not None:
        content = [{"type": "image", "image": image_ref}, *content]

    return [
        segment
        for segment in content
        if segment["type"] != "text"
        or (segment.get("text", "") and segment["text"].strip())
    ]


def _normalize_llava_instruct(example: dict) -> dict:
    conversations = example.get("conversations") or []
    image_ref = _normalize_media_ref(example.get("image"))

    normalized_conversations = []
    image_attached = False
    for turn in conversations:
        role = turn.get("from", turn.get("role"))
        value = turn.get("value", turn.get("content", ""))
        if role in {"human", "user"} and image_ref is not None:
            content = _split_llava_user_content(str(value), image_ref)
            image_attached = image_attached or any(
                seg.get("type") == "image" for seg in content
            )
            normalized_conversations.append({"from": role, "value": content})
        else:
            normalized_conversations.append(turn)

    if image_ref is not None and not image_attached:
        for turn in normalized_conversations:
            role = turn.get("from", turn.get("role"))
            if role in {"human", "user"}:
                text = turn.get("value", turn.get("content", ""))
                text = text if isinstance(text, str) else ""
                turn["value"] = _split_llava_user_content(text, image_ref)
                break

    return {"conversations": normalized_conversations}


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
    ),
    "llava-instruct": DatasetConfig(
        name="llava-instruct",
        hf_path="liuhaotian/LLaVA-Instruct-150K",
        split="train",
        normalize_fn=_normalize_llava_instruct,
    ),
}
