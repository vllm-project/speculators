"""Safely materialize embedded Parquet images for multimodal training.

The Qwen3-VL online examples download Parquet shards whose ``image`` column
contains embedded bytes.  This helper writes those bytes below a dedicated
``materialized_images`` directory and emits the JSONL consumed by
``prepare_data.py``.

Only relative, local source references are accepted.  Each row must contain
exactly one supported image content part, and that reference must agree with
the top-level image path metadata when such metadata is present.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit


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
    """Raised when input data violates the materialization safety contract."""


def _fsync_directory(directory: Path) -> None:
    """Durably commit a directory entry change on the local filesystem."""
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max samples must be a positive integer",
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max samples must be a positive integer")
    return parsed


def _reject_symlink_components(root: Path, relative_path: Path, *, field: str) -> None:
    current = root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            raise MaterializationError(f"{field} traverses symlink: {relative_path}")


def _validate_relative_reference(
    reference: object,
    *,
    dataset_dir: Path,
    field: str,
) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise MaterializationError(f"{field} must be a non-empty string path")
    if "\x00" in reference or "\\" in reference:
        raise MaterializationError(
            f"{field} is not a safe POSIX relative path: {reference}",
        )

    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise MaterializationError(f"{field} must be local and relative: {reference}")

    posix_path = PurePosixPath(reference)
    windows_path = PureWindowsPath(reference)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise MaterializationError(f"{field} must not be absolute: {reference}")
    if any(part == ".." for part in posix_path.parts):
        raise MaterializationError(
            f"{field} escapes the dataset directory: {reference}",
        )
    if not posix_path.parts:
        raise MaterializationError(f"{field} must name an image file")

    relative_path = Path(*posix_path.parts)
    _reject_symlink_components(dataset_dir, relative_path, field=field)
    resolved = (dataset_dir / relative_path).resolve(strict=False)
    if not resolved.is_relative_to(dataset_dir):
        raise MaterializationError(
            f"{field} escapes the dataset directory: {reference}",
        )
    return relative_path


def _extract_image_part_reference(part: dict[str, Any], *, field: str) -> object:
    part_type = part.get("type")
    if part_type not in SUPPORTED_IMAGE_PART_TYPES:
        raise MaterializationError(f"{field} has unsupported image type: {part_type!r}")

    candidates: list[object] = []
    for key in ("image", "path", "url"):
        if part.get(key) is not None:
            candidates.append(part[key])

    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        if image_url.get("url") is not None:
            candidates.append(image_url["url"])
    elif image_url is not None:
        candidates.append(image_url)

    if len(candidates) != 1:
        raise MaterializationError(
            f"{field} must contain exactly one unambiguous image reference",
        )
    return candidates[0]


def _find_single_image_part(
    sample: dict[str, Any],
    *,
    row_idx: int,
) -> tuple[dict[str, Any], object]:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise MaterializationError(
            f"row {row_idx}: conversations must be a non-empty list",
        )

    matches: list[tuple[dict[str, Any], object]] = []
    for turn_idx, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            raise MaterializationError(
                f"row {row_idx}: turn {turn_idx} must be an object",
            )
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in SUPPORTED_IMAGE_PART_TYPES:
                continue
            field = f"row {row_idx} turn {turn_idx} part {part_idx}"
            matches.append((part, _extract_image_part_reference(part, field=field)))

    if len(matches) != 1:
        raise MaterializationError(
            f"row {row_idx}: expected exactly one supported image part, "
            f"found {len(matches)}",
        )
    return matches[0]


def _embedded_image_bytes_and_paths(
    sample: dict[str, Any],
    *,
    row_idx: int,
) -> tuple[bytes, object | None, object | None]:
    image = sample.get("image")
    if not isinstance(image, dict):
        raise MaterializationError(f"row {row_idx}: image must contain embedded bytes")

    image_bytes = image.get("bytes")
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or not image_bytes:
        raise MaterializationError(f"row {row_idx}: image.bytes must be non-empty")

    return bytes(image_bytes), sample.get("image_path"), image.get("path")


def _validate_embedded_storage_path(
    reference: object,
    *,
    conversation_path: Path,
    dataset_dir: Path,
    row_idx: int,
) -> None:
    """Validate HF Image storage metadata without treating it as authoritative.

    Hugging Face ``Image.embed_storage`` keeps embedded bytes but normalizes the
    storage ``path`` to ``basename(original_path)``.  The dataset's explicit
    ``image_path`` and conversation image reference retain the full relative path,
    so those two fields are authoritative.  The storage hint may either match that
    path exactly or be its basename only; no other path form is accepted.
    """
    if reference is None:
        return
    storage_path = _validate_relative_reference(
        reference,
        dataset_dir=dataset_dir,
        field=f"row {row_idx} image.path",
    )
    is_basename_hint = (
        storage_path.parent == Path(".")
        and storage_path.name == conversation_path.name
    )
    if storage_path != conversation_path and not is_basename_hint:
        raise MaterializationError(
            f"row {row_idx}: embedded image.path {storage_path} neither matches "
            f"the conversation path {conversation_path} nor its basename",
        )


def _safe_image_suffix(relative_path: Path, *, field: str) -> str:
    suffix = relative_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise MaterializationError(f"{field} has unsupported image suffix: {suffix!r}")
    return suffix


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise MaterializationError(f"refusing to replace symlink: {path}")
    if path.exists():
        if not path.is_file():
            raise MaterializationError(f"materialized image path is not a file: {path}")
        existing_hash = hashlib.sha256(path.read_bytes()).digest()
        if existing_hash != hashlib.sha256(payload).digest():
            raise MaterializationError(
                f"existing content-addressed image is corrupt: {path}",
            )
        return

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _prepare_output_directory(dataset_dir: Path) -> Path:
    output_dir = dataset_dir / MATERIALIZED_IMAGES_DIRNAME
    if output_dir.is_symlink():
        raise MaterializationError(
            f"materialized image directory is a symlink: {output_dir}",
        )
    existed = output_dir.exists()
    output_dir.mkdir(parents=False, exist_ok=True)
    if not output_dir.is_dir() or output_dir.resolve() != output_dir:
        raise MaterializationError(f"unsafe materialized image directory: {output_dir}")
    if not existed:
        _fsync_directory(dataset_dir)
    return output_dir


def _materialize_row(
    raw_sample: Mapping[str, Any],
    *,
    row_idx: int,
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    sample = copy.deepcopy(dict(raw_sample))
    image_part, conversation_reference = _find_single_image_part(
        sample,
        row_idx=row_idx,
    )
    (
        image_bytes,
        image_path_reference,
        storage_path_reference,
    ) = _embedded_image_bytes_and_paths(
        sample,
        row_idx=row_idx,
    )

    conversation_path = _validate_relative_reference(
        conversation_reference,
        dataset_dir=dataset_dir,
        field=f"row {row_idx} conversation image",
    )
    if image_path_reference is not None:
        top_level_path = _validate_relative_reference(
            image_path_reference,
            dataset_dir=dataset_dir,
            field=f"row {row_idx} image_path",
        )
        if top_level_path != conversation_path:
            raise MaterializationError(
                f"row {row_idx}: top-level image path {top_level_path} does not match "
                f"conversation image path {conversation_path}",
            )
    _validate_embedded_storage_path(
        storage_path_reference,
        conversation_path=conversation_path,
        dataset_dir=dataset_dir,
        row_idx=row_idx,
    )

    suffix = _safe_image_suffix(
        conversation_path,
        field=f"row {row_idx} conversation image",
    )
    content_hash = hashlib.sha256(image_bytes).hexdigest()
    image_path = output_dir / f"{row_idx:08d}-{content_hash}{suffix}"
    _atomic_write_bytes(image_path, image_bytes)
    absolute_image_path = str(image_path)

    for key in ("image", "path", "url", "image_url"):
        image_part.pop(key, None)
    image_part["type"] = "image"
    image_part["image"] = absolute_image_path
    sample["image"] = absolute_image_path
    sample.pop("image_path", None)
    return sample, image_path


def _validate_materialized_image_set(
    output_dir: Path,
    expected_files: set[Path],
) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise MaterializationError(f"unsafe materialized image directory: {output_dir}")

    actual_files: set[Path] = set()
    for entry in output_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise MaterializationError(
                f"materialized image directory contains a non-regular file: {entry}",
            )
        actual_files.add(entry)

    if actual_files != expected_files:
        extra = sorted(path.name for path in actual_files - expected_files)
        missing = sorted(path.name for path in expected_files - actual_files)
        raise MaterializationError(
            "materialized image set does not match this run; "
            f"extra={extra}, missing={missing}. Use a fresh dataset directory.",
        )


def materialize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_dir: Path,
    max_samples: int,
) -> tuple[Path, int]:
    """Materialize exactly ``max_samples`` rows and atomically replace the JSONL."""
    if max_samples <= 0:
        raise MaterializationError("max_samples must be a positive integer")
    if dataset_dir.is_symlink():
        raise MaterializationError(f"dataset directory is a symlink: {dataset_dir}")
    dataset_dir = dataset_dir.resolve(strict=True)
    if not dataset_dir.is_dir():
        raise MaterializationError(
            f"dataset directory is not a directory: {dataset_dir}",
        )

    output_dir = _prepare_output_directory(dataset_dir)
    destination = dataset_dir / OUTPUT_JSONL_FILENAME
    if destination.is_symlink():
        raise MaterializationError(f"output JSONL is a symlink: {destination}")

    fd, tmp_name = tempfile.mkstemp(
        dir=dataset_dir,
        prefix=f".{OUTPUT_JSONL_FILENAME}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    count = 0
    expected_image_files: set[Path] = set()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row_idx, raw_sample in enumerate(islice(rows, max_samples)):
                sample, image_path = _materialize_row(
                    raw_sample,
                    row_idx=row_idx,
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                )
                expected_image_files.add(image_path)
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
            if count != max_samples:
                raise MaterializationError(
                    f"expected {max_samples} rows, but the input only provided {count}",
                )
            output.flush()
            os.fsync(output.fileno())
        _validate_materialized_image_set(output_dir, expected_image_files)
        os.replace(tmp_path, destination)
        _fsync_directory(dataset_dir)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return destination, count


def _load_parquet_rows(dataset_dir: Path):
    parquet_dir = dataset_dir / "data"
    if parquet_dir.is_symlink():
        raise MaterializationError(f"Parquet directory is a symlink: {parquet_dir}")
    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found under {parquet_dir}")
    for parquet_file in parquet_files:
        if parquet_file.is_symlink() or not parquet_file.is_file():
            raise MaterializationError(f"unsafe Parquet shard: {parquet_file}")

    from datasets import Image, load_dataset  # noqa: PLC0415

    return load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
    ).cast_column("image", Image(decode=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely materialize embedded multimodal Parquet images",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=_positive_int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    if dataset_dir.is_symlink():
        raise MaterializationError(f"dataset directory is a symlink: {dataset_dir}")
    dataset_dir = dataset_dir.resolve(strict=True)
    destination, count = materialize_rows(
        _load_parquet_rows(dataset_dir),
        dataset_dir=dataset_dir,
        max_samples=args.max_samples,
    )
    print(f"Wrote {count} rows to {destination}")


if __name__ == "__main__":
    main()
