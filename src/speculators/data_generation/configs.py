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
    filter_fn: Callable[[dict], bool] | None = None
    normalize_fn: Callable[[dict], dict] | None = None


def get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


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
        normalize_fn=_normalize_ultrachat,
    ),
    # NOTE: You need to pass `--allowed-local-media-path /` to `launch_vllm.py`
    "sharegpt4v_coco": DatasetConfig(
        name="sharegpt4v_coco",
        hf_path="Lin-Chen/ShareGPT4V",
        hf_name="ShareGPT4V",
        split="train",
        filter_fn=_filter_sharegpt4v_coco,
        normalize_fn=_normalize_sharegpt4v_coco,
    ),
}
