"""Materialize embedded Parquet images for multimodal training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

MATERIALIZED_IMAGES_DIRNAME = "materialized_images"
OUTPUT_JSONL_FILENAME = "train.absolute_paths.jsonl"
SUPPORTED_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}
SUPPORTED_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class MaterializationError(ValueError):
    """Raised when a row cannot be safely materialized."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _relative_image_path(reference: object, *, field: str) -> PurePosixPath:
    if not isinstance(reference, str) or not reference.strip():
        raise MaterializationError(f"{field} must be a non-empty path")
    if "\\" in reference or "\x00" in reference:
        raise MaterializationError(f"{field} must be a POSIX relative path")

    parsed = urlsplit(reference)
    path = PurePosixPath(reference)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise MaterializationError(f"{field} must be a local relative path")
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise MaterializationError(f"{field} has an unsupported image suffix")
    return path


def _part_reference(part: dict[str, Any], *, field: str) -> object:
    candidates = [
        part[key] for key in ("image", "path", "url") if part.get(key) is not None
    ]
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        if "url" in image_url:
            candidates.append(image_url["url"])
    elif image_url is not None:
        candidates.append(image_url)
    if len(candidates) != 1:
        raise MaterializationError(f"{field} has an ambiguous image reference")
    return candidates[0]


def _single_image_part(
    sample: dict[str, Any],
    *,
    row_idx: int,
) -> tuple[dict[str, Any], PurePosixPath]:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise MaterializationError(f"row {row_idx}: conversations must be a list")

    matches: list[tuple[dict[str, Any], object]] = []
    for turn in conversations:
        if not isinstance(turn, dict) or not isinstance(turn.get("content"), list):
            continue
        for part in turn["content"]:
            if (
                isinstance(part, dict)
                and part.get("type") in SUPPORTED_IMAGE_PART_TYPES
            ):
                matches.append((part, _part_reference(part, field=f"row {row_idx}")))

    if len(matches) != 1:
        raise MaterializationError(
            f"row {row_idx}: expected exactly one image part, found {len(matches)}",
        )
    part, reference = matches[0]
    return part, _relative_image_path(reference, field=f"row {row_idx} image")


def _embedded_bytes(sample: dict[str, Any], *, row_idx: int) -> tuple[bytes, object]:
    image = sample.get("image")
    if not isinstance(image, dict):
        raise MaterializationError(f"row {row_idx}: image must contain embedded bytes")
    payload = image.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise MaterializationError(f"row {row_idx}: image.bytes must be non-empty")
    return bytes(payload), image.get("path")


def _validate_path_metadata(
    sample: dict[str, Any],
    *,
    conversation_path: PurePosixPath,
    storage_reference: object,
    row_idx: int,
) -> None:
    top_level_reference = sample.get("image_path")
    if top_level_reference is not None:
        top_level_path = _relative_image_path(
            top_level_reference,
            field=f"row {row_idx} image_path",
        )
        if top_level_path != conversation_path:
            raise MaterializationError(
                f"row {row_idx}: image_path does not match the conversation image",
            )

    if storage_reference is None:
        return
    storage_path = _relative_image_path(
        storage_reference,
        field=f"row {row_idx} image.path",
    )
    # datasets.Image.embed_storage may reduce the stored path to its basename.
    if storage_path != conversation_path and storage_path != PurePosixPath(
        conversation_path.name,
    ):
        raise MaterializationError(
            f"row {row_idx}: image.path does not match the conversation image",
        )


def _prepare_output_directory(dataset_dir: Path) -> Path:
    output_dir = dataset_dir / MATERIALIZED_IMAGES_DIRNAME
    if output_dir.is_symlink():
        raise MaterializationError(f"output directory is a symlink: {output_dir}")
    output_dir.mkdir(exist_ok=True)
    if not output_dir.is_dir():
        raise MaterializationError(f"output path is not a directory: {output_dir}")
    return output_dir


def _materialize_row(
    raw_sample: Mapping[str, Any],
    *,
    row_idx: int,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    sample = copy.deepcopy(dict(raw_sample))
    image_part, conversation_path = _single_image_part(sample, row_idx=row_idx)
    payload, storage_reference = _embedded_bytes(sample, row_idx=row_idx)
    _validate_path_metadata(
        sample,
        conversation_path=conversation_path,
        storage_reference=storage_reference,
        row_idx=row_idx,
    )

    digest = hashlib.sha256(payload).hexdigest()
    filename = f"{row_idx:08d}-{digest}{conversation_path.suffix.lower()}"
    image_path = output_dir / filename
    if image_path.is_symlink():
        raise MaterializationError(f"image output is a symlink: {image_path}")
    image_path.write_bytes(payload)

    for key in ("image", "path", "url", "image_url"):
        image_part.pop(key, None)
    image_part.update(type="image", image=str(image_path))
    sample["image"] = str(image_path)
    sample.pop("image_path", None)
    return sample, image_path


def _remove_stale_images(output_dir: Path, expected: set[Path]) -> None:
    for entry in output_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise MaterializationError(f"unexpected output entry: {entry}")
        if entry not in expected:
            entry.unlink()


def materialize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_dir: Path,
    max_samples: int,
) -> tuple[Path, int]:
    """Materialize ``max_samples`` rows and atomically replace the JSONL."""
    if max_samples <= 0:
        raise MaterializationError("max_samples must be positive")
    if dataset_dir.is_symlink():
        raise MaterializationError(f"dataset directory is a symlink: {dataset_dir}")
    dataset_dir = dataset_dir.resolve(strict=True)
    if not dataset_dir.is_dir():
        raise MaterializationError(f"not a dataset directory: {dataset_dir}")

    output_dir = _prepare_output_directory(dataset_dir)
    destination = dataset_dir / OUTPUT_JSONL_FILENAME
    if destination.is_symlink():
        raise MaterializationError(f"output JSONL is a symlink: {destination}")

    fd, tmp_name = tempfile.mkstemp(dir=dataset_dir, suffix=".jsonl.tmp")
    tmp_path = Path(tmp_name)
    expected_images: set[Path] = set()
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row_idx, row in enumerate(islice(rows, max_samples)):
                sample, image_path = _materialize_row(
                    row,
                    row_idx=row_idx,
                    output_dir=output_dir,
                )
                expected_images.add(image_path)
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
        if count != max_samples:
            raise MaterializationError(
                f"expected {max_samples} rows, but the input provided {count}",
            )
        os.replace(tmp_path, destination)
        _remove_stale_images(output_dir, expected_images)
    finally:
        tmp_path.unlink(missing_ok=True)

    return destination, count


def _load_parquet_rows(dataset_dir: Path):
    parquet_dir = dataset_dir / "data"
    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no Parquet shards found under {parquet_dir}")
    if parquet_dir.is_symlink() or any(
        path.is_symlink() or not path.is_file() for path in parquet_files
    ):
        raise MaterializationError(f"unsafe Parquet input under {parquet_dir}")

    from datasets import Image, load_dataset  # noqa: PLC0415

    return load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
    ).cast_column("image", Image(decode=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize embedded multimodal Parquet images",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=_positive_int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    destination, count = materialize_rows(
        _load_parquet_rows(dataset_dir),
        dataset_dir=dataset_dir,
        max_samples=args.max_samples,
    )
    print(f"Wrote {count} rows to {destination}")


if __name__ == "__main__":
    main()
