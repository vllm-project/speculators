import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import materialize_multimodal_parquet as materializer


def _row(
    image_path: str,
    image_bytes: bytes = b"embedded-image",
    *,
    image_part: dict | None = None,
    embedded_path: str | None = None,
) -> dict:
    image_part = image_part or {"type": "image", "image": image_path}
    return {
        "image": {
            "bytes": image_bytes,
            "path": image_path if embedded_path is None else embedded_path,
        },
        "image_path": image_path,
        "conversations": [
            {
                "from": "human",
                "content": [
                    image_part,
                    {"type": "text", "text": "Describe the image."},
                ],
            },
            {"from": "gpt", "content": "A test image."},
        ],
    }


def _make_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    return dataset_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_materializes_canonical_absolute_image_path(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    payload = b"test-jpeg-bytes"
    source_row = _row(
        "images/example.JPG",
        payload,
        image_part={
            "type": "image_url",
            "image_url": {"url": "images/example.JPG"},
            "detail": "high",
        },
    )

    destination, count = materializer.materialize_rows(
        [source_row],
        dataset_dir=dataset_dir,
        max_samples=1,
    )

    expected_image = (
        dataset_dir
        / materializer.MATERIALIZED_IMAGES_DIRNAME
        / f"00000000-{hashlib.sha256(payload).hexdigest()}.jpg"
    ).resolve()
    written = _read_jsonl(destination)
    image_part = written[0]["conversations"][0]["content"][0]

    assert count == 1
    assert expected_image.read_bytes() == payload
    assert written[0]["image"] == str(expected_image)
    assert "image_path" not in written[0]
    assert image_part == {
        "type": "image",
        "image": str(expected_image),
        "detail": "high",
    }
    assert isinstance(source_row["image"], dict), "the caller's row must not be mutated"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.jpg",
        "/absolute/image.jpg",
        "https://example.com/image.jpg",
        "file:///tmp/image.jpg",
        "C:\\images\\image.jpg",
    ],
)
def test_rejects_escaping_absolute_and_remote_references(
    tmp_path: Path,
    unsafe_path: str,
):
    dataset_dir = _make_dataset_dir(tmp_path)

    with pytest.raises(materializer.MaterializationError):
        materializer.materialize_rows(
            [_row(unsafe_path)],
            dataset_dir=dataset_dir,
            max_samples=1,
        )


def test_rejects_symlink_in_original_reference(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (dataset_dir / "images").symlink_to(outside, target_is_directory=True)

    with pytest.raises(materializer.MaterializationError, match="traverses symlink"):
        materializer.materialize_rows(
            [_row("images/example.jpg")],
            dataset_dir=dataset_dir,
            max_samples=1,
        )


def test_rejects_multiple_conversation_images(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    row = _row("images/first.jpg")
    row["conversations"][0]["content"].append(
        {"type": "image", "image": "images/second.jpg"},
    )

    with pytest.raises(materializer.MaterializationError, match="exactly one"):
        materializer.materialize_rows(
            [row],
            dataset_dir=dataset_dir,
            max_samples=1,
        )


def test_rejects_top_level_and_conversation_path_mismatch(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    row = _row(
        "images/top-level.jpg",
        image_part={"type": "image", "image": "images/conversation.jpg"},
    )

    with pytest.raises(materializer.MaterializationError, match="does not match"):
        materializer.materialize_rows(
            [row],
            dataset_dir=dataset_dir,
            max_samples=1,
        )


def test_accepts_hf_embedded_storage_basename(tmp_path: Path):
    """HF Image.embed_storage reduces image.path to the original basename."""
    dataset_dir = _make_dataset_dir(tmp_path)
    source_path = "images/textvqa/train_images/929b48d1323c8323.jpg"

    destination, count = materializer.materialize_rows(
        [_row(source_path, embedded_path="929b48d1323c8323.jpg")],
        dataset_dir=dataset_dir,
        max_samples=1,
    )

    assert count == 1
    image_reference = _read_jsonl(destination)[0]["conversations"][0]["content"][
        0
    ]["image"]
    assert image_reference.startswith(str(dataset_dir / "materialized_images"))


@pytest.mark.parametrize(
    "embedded_path",
    ["different.jpg", "other/929b48d1323c8323.jpg"],
)
def test_rejects_unrelated_embedded_storage_path(
    tmp_path: Path,
    embedded_path: str,
):
    dataset_dir = _make_dataset_dir(tmp_path)

    with pytest.raises(materializer.MaterializationError, match="image.path"):
        materializer.materialize_rows(
            [
                _row(
                    "images/textvqa/train_images/929b48d1323c8323.jpg",
                    embedded_path=embedded_path,
                )
            ],
            dataset_dir=dataset_dir,
            max_samples=1,
        )


def test_jsonl_is_not_replaced_when_a_later_row_fails(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    destination = dataset_dir / materializer.OUTPUT_JSONL_FILENAME
    destination.write_text("previous-complete-output\n")
    invalid_row = _row(
        "images/top-level.jpg",
        image_part={"type": "image", "image": "images/mismatch.jpg"},
    )

    with pytest.raises(materializer.MaterializationError):
        materializer.materialize_rows(
            [_row("images/valid.jpg"), invalid_row],
            dataset_dir=dataset_dir,
            max_samples=2,
        )

    assert destination.read_text() == "previous-complete-output\n"
    assert not list(dataset_dir.glob(".*.tmp"))
    materialized_dir = dataset_dir / materializer.MATERIALIZED_IMAGES_DIRNAME
    assert not list(materialized_dir.glob(".*.tmp"))


def test_rejects_stale_extra_materialized_image_before_jsonl_replace(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    destination = dataset_dir / materializer.OUTPUT_JSONL_FILENAME
    destination.write_text("previous-complete-output\n")
    materialized_dir = dataset_dir / materializer.MATERIALIZED_IMAGES_DIRNAME
    materialized_dir.mkdir()
    stale_file = materialized_dir / "stale-old-revision.jpg"
    stale_file.write_bytes(b"stale")

    with pytest.raises(
        materializer.MaterializationError,
        match="does not match this run",
    ):
        materializer.materialize_rows(
            [_row("images/current.jpg")],
            dataset_dir=dataset_dir,
            max_samples=1,
        )

    assert destination.read_text() == "previous-complete-output\n"
    assert stale_file.read_bytes() == b"stale"
    assert not list(dataset_dir.glob(".*.tmp"))


def test_images_and_jsonl_are_committed_with_os_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset_dir = _make_dataset_dir(tmp_path)
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(materializer.os, "replace", recording_replace)

    destination, _ = materializer.materialize_rows(
        [_row("images/example.png", b"png-bytes")],
        dataset_dir=dataset_dir,
        max_samples=1,
    )

    assert len(replace_calls) == 2
    assert all(source.suffix == ".tmp" for source, _ in replace_calls)
    assert replace_calls[0][1].parent.name == materializer.MATERIALIZED_IMAGES_DIRNAME
    assert replace_calls[1][1] == destination


def test_image_and_jsonl_directory_entries_are_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset_dir = _make_dataset_dir(tmp_path)
    fsynced: list[Path] = []
    monkeypatch.setattr(
        materializer,
        "_fsync_directory",
        lambda path: fsynced.append(Path(path).resolve()),
    )

    materializer.materialize_rows(
        [_row("images/example.png", b"png-bytes")],
        dataset_dir=dataset_dir,
        max_samples=1,
    )

    assert (dataset_dir / materializer.MATERIALIZED_IMAGES_DIRNAME).resolve() in fsynced
    # Creating the image directory and replacing the JSONL both durably update
    # the dataset directory; at least one fsync must survive future refactors.
    assert dataset_dir.resolve() in fsynced


def test_max_samples_controls_output_row_count(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    rows = [
        _row(f"images/{idx}.jpg", f"image-{idx}".encode())
        for idx in range(3)
    ]

    destination, count = materializer.materialize_rows(
        rows,
        dataset_dir=dataset_dir,
        max_samples=2,
    )

    assert count == 2
    assert len(_read_jsonl(destination)) == 2
    assert len(list((dataset_dir / "materialized_images").iterdir())) == 2


def test_rejects_insufficient_rows_without_replacing_jsonl(tmp_path: Path):
    dataset_dir = _make_dataset_dir(tmp_path)
    destination = dataset_dir / materializer.OUTPUT_JSONL_FILENAME
    destination.write_text("previous-complete-output\n")

    with pytest.raises(materializer.MaterializationError, match="expected 3 rows"):
        materializer.materialize_rows(
            [_row("images/zero.jpg"), _row("images/one.jpg")],
            dataset_dir=dataset_dir,
            max_samples=3,
        )

    assert destination.read_text() == "previous-complete-output\n"
    assert not list(dataset_dir.glob(".*.tmp"))


@pytest.mark.parametrize("value", ["0", "-1", "all", "none", ""])
def test_max_samples_rejects_non_positive_and_sentinel_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        materializer._positive_int(value)
