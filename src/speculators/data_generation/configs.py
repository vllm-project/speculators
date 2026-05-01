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


def hf_to_vllm_part(part: str | dict):
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]

    if part_type == "text":
        return {"type": "text", "text": part["text"]}

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

    hf_messages = [
        {
            "content": [
                _unformat_sharegpt4v(part, image_path)
                for part in turn.pop("value").split("\n")
            ],
            **turn,
        }
        for turn in example["conversations"]
    ]
    vllm_messages = [
        {
            "content": [hf_to_vllm_part(part) for part in turn.pop("content")],
            **turn,
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
    "sharegpt4v_coco": DatasetConfig(
        name="sharegpt4v_coco",
        hf_path="Lin-Chen/ShareGPT4V",
        hf_name="ShareGPT4V",
        split="train",
        filter_fn=_filter_sharegpt4v_coco,
        normalize_fn=_normalize_sharegpt4v_coco,
    ),
}
