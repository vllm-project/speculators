"""Configuration registries for data generation pipeline."""

from collections.abc import Callable
from dataclasses import dataclass

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


def _normalize_coco(example: dict) -> dict:
    pil_image = example["image"]
    task = "Describe the image with a brief caption."
    caption = example["sentences"]["raw"]

    hf_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": pil_image.filename,
                },
                {
                    "type": "text",
                    "text": task,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": caption,
                }
            ]
        },
    ]
    vllm_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"file://{pil_image.filename}",
                },
                {
                    "type": "text",
                    "text": task,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": caption,
                }
            ]
        },
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
    # NOTE: `datasets<4` is needed to run custom script
    # You also need to pass `--allowed-local-media-path /` to `launch_vllm.py`
    "coco": DatasetConfig(
        name="coco",
        hf_path="HuggingFaceM4/COCO",
        split="train",
        normalize_fn=_normalize_coco,
    ),
}
