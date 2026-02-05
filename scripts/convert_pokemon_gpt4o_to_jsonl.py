#!/usr/bin/env python3
"""Convert a HF dataset with images + conversations into speculators JSONL.

This script loads a HuggingFace dataset split, attaches each sample's images to the
first user turn, and writes a JSONL file with speculators-style multimodal messages.

Example:
    python scripts/convert_pokemon_gpt4o_to_jsonl.py \
    --dataset llamafactory/pokemon-gpt4o-captions \
    --split train \
    --output ./data/pokemon_vl.jsonl \
    --image-output-dir ./pokemon_images

"""

import argparse
import json
import logging
import os
import re
from io import BytesIO
from typing import Any

from datasets import Image, Sequence, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for dataset conversion to JSONL.
    parser = argparse.ArgumentParser(
        description="Convert pokemon-gpt4o-captions to JSONL with multimodal content."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., llamafactory/pokemon-gpt4o-captions).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help=(
            "Optional root directory for resolving image paths. If set, image"
            " paths are written as basenames and resolved later using --image-root"
            " in data generation."
        ),
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write image bytes when the dataset stores images as bytes"
            " (e.g., decode=False). If omitted, byte images will raise an error."
        ),
    )
    return parser.parse_args()


def _normalize_role(role: str) -> str:
    # Normalize dataset roles to speculators roles.
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant"):
        return "assistant"
    return role


def _normalize_image_path(image_path: str, image_root: str | None) -> str:
    # Normalize image paths and apply optional root prefix.
    if image_path.startswith("file://"):
        image_path = image_path[len("file://") :]
    if image_root:
        return os.path.basename(image_path)
    if (not image_path.startswith(("http://", "https://", "data:"))) and (
        not os.path.isabs(image_path)
    ):
        return image_path
    return image_path


def _write_image_bytes(
    image_bytes: bytes | memoryview,
    output_dir: str,
    sample_idx: int,
    image_idx: int,
) -> str:
    # Save raw image bytes to disk and return the saved file path.
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    os.makedirs(output_dir, exist_ok=True)
    image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
    filename = f"img_{sample_idx}_{image_idx}.png"
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    return filename


def _copy_image_file(
    image_path: str,
    output_dir: str,
    sample_idx: int,
    image_idx: int,
) -> str:
    # Copy an image file into the output directory and return the saved filename.
    os.makedirs(output_dir, exist_ok=True)
    image = PILImage.open(image_path).convert("RGB")
    filename = f"img_{sample_idx}_{image_idx}.png"
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    return filename


def _prepare_image_ref(
    image_path: str,
    image_root: str | None,
    image_output_dir: str | None,
    sample_idx: int,
    image_idx: int,
) -> str:
    # Normalize and optionally copy image paths into the output directory.
    if image_path.startswith("file://"):
        image_path = image_path[len("file://") :]
    if image_output_dir:
        if image_path.startswith(("http://", "https://", "data:")):
            return image_path
        if image_root and not os.path.isabs(image_path):
            image_path = os.path.join(image_root, image_path)
        return _copy_image_file(image_path, image_output_dir, sample_idx, image_idx)
    return _normalize_image_path(image_path, image_root)


def _resolve_image_ref(
    image_ref: Any,
    image_root: str | None,
    image_output_dir: str | None,
    sample_idx: int,
    image_idx: int,
) -> str:
    # Resolve image references into file paths or URLs.
    if isinstance(image_ref, dict):
        path = image_ref.get("path") or image_ref.get("image")
        url = image_ref.get("url")
        if path:
            return _prepare_image_ref(
                path, image_root, image_output_dir, sample_idx, image_idx
            )
        if url:
            return url
        image_bytes = image_ref.get("bytes")
        if image_bytes is not None:
            if image_output_dir is None:
                raise ValueError(
                    "Image bytes found but --image-output-dir was not provided."
                )
            return _write_image_bytes(
                image_bytes, image_output_dir, sample_idx, image_idx
            )
    elif isinstance(image_ref, str):
        return _prepare_image_ref(
            image_ref, image_root, image_output_dir, sample_idx, image_idx
        )
    raise ValueError(f"Unsupported image reference: {type(image_ref)}")


def _extract_image_refs(
    images: list[Any],
    image_root: str | None,
    image_output_dir: str | None,
    sample_idx: int,
) -> list[str]:
    # Extract all image references for a sample into a list of strings.
    image_refs: list[str] = []
    for image_idx, image_ref in enumerate(images):
        image_refs.append(
            _resolve_image_ref(
                image_ref, image_root, image_output_dir, sample_idx, image_idx
            )
        )
    return image_refs


def _build_messages(
    conversations: list[dict[str, Any]],
    image_refs: list[str],
) -> list[dict[str, Any]]:
    # Build speculators-style messages with images attached to first user turn.
    messages: list[dict[str, Any]] = []
    images_attached = False
    for turn in conversations:
        role = _normalize_role(turn.get("from") or turn.get("role", ""))
        text = turn.get("value") or turn.get("content") or ""
        text = re.sub(r"<\s*/?\s*image\s*>", "", text, flags=re.IGNORECASE).strip()
        if role == "user" and image_refs and not images_attached:
            content: list[dict[str, Any]] = [
                {"type": "image", "image": ref} for ref in image_refs
            ]
            content.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content})
            images_attached = True
            continue
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    return messages


def main() -> None:
    # Run dataset conversion from HF/parquet to JSONL.
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    image_root = args.image_root
    image_output_dir = args.image_output_dir

    if not args.dataset:
        raise ValueError("Provide --dataset.")

    logger.info("Loading dataset...")
    ds = load_dataset(args.dataset, split=args.split)

    if "images" in ds.column_names:
        ds = ds.cast_column("images", Sequence(Image(decode=False)))

    logger.info("Loaded %s samples. Writing JSONL to %s", len(ds), args.output)
    total = 0
    with open(args.output, "w") as f:
        for idx, row in enumerate(tqdm(ds, desc="Converting", unit="sample")):
            conversations = row.get("conversations") or row.get("messages")
            if not conversations:
                continue
            images = row.get("images") or []
            image_refs = _extract_image_refs(images, image_root, image_output_dir, idx)
            messages = _build_messages(conversations, image_refs)
            out = {"conversations": messages}
            if "lang" in row:
                out["lang"] = row["lang"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1
    logger.info("Wrote %s samples to %s", total, args.output)


if __name__ == "__main__":
    main()
