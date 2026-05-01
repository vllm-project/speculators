"""Configuration registries for data generation pipeline."""

import os
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for loading a dataset"""

    name: str
    hf_path: str
    hf_name: str | None = None
    split: str
    normalize_fn: Callable[[dict], dict] | None = None


def hf_to_vllm_part(part: str | dict):
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]

    for modality in ("image", "video", "audio"):
        if part_type == modality:
            if local_path := part.get("path"):
                file_url = f"file://{local_path}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}
            if url := part.get("url"):
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}

            fields_expr = {f"part.{k}" for k in part if k != "type"}

            raise NotImplementedError(
                f"No handler defined in part.type={part_type!r} "
                f"for fields: {fields_expr}"
            )

    raise NotImplementedError(f"No handler defined for part.type={part_type!r}")


def get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


def _unformat_sharegpt4v(part: str, image_path: str):
    if part == "<image>":
        return {"type": "image", "path": image_path}

    return {"type": "text", "text": part}


def _normalize_sharegpt4v(example: dict) -> dict:
    image_path: str = example["image"]
    image_path = os.path.join(get_coco_dir(), image_path.removeprefix("coco/"))

    if not os.path.exists(image_path):
        raise ValueError(
            "Please download COCO 2017 Train Images from "
            "http://images.cocodataset.org/zips/train2017.zip and "
            "place the files under `COCO_DIR` (default: `./coco`)."
        )

    hf_messages = [
        {
            **turn,
            "content": [
                _unformat_sharegpt4v(part, image_path)
                for part in turn.pop("value").split("\n")
            ],
        }
        for turn in example["conversations"]
    ]
    vllm_messages = [
        {
            **turn,
            "content": [hf_to_vllm_part(part) for part in turn["content"]],
        }
        for turn in hf_messages
    ]

    return {"conversations": hf_messages, "_vllm_messages": vllm_messages}


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
    # NOTE: You need to pass `--allowed-local-media-path /` to `launch_vllm.py`
    "sharegpt4v": DatasetConfig(
        name="sharegpt4v",
        hf_path="Lin-Chen/ShareGPT4V",
        hf_name="ShareGPT4V",
        split="train",
        normalize_fn=_normalize_sharegpt4v,
    ),
}
