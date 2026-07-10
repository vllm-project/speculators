#!/usr/bin/env python3
"""Guard example pipelines against stale or mixed reusable artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import numbers
import os
import pickle
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA_VERSION = 2
_FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_FIELD_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TRAIN_SHARD_RE = re.compile(r"^train-(\d+)-of-(\d+)\.parquet$")
_TEMPORARY_FILE_SUFFIXES = (".tmp", ".temp", ".partial", ".incomplete")
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}
_UNSUPPORTED_MEDIA_PART_TYPES = {
    "audio",
    "audio_url",
    "input_audio",
    "input_video",
    "video",
    "video_url",
}


def _validate_inactive_commit_lock(lock_path: Path) -> None:
    """Accept a persistent cache lock only when it is empty and not held.

    vLLM may return a noncanonical staging filename.  Deleting that source
    intentionally leaves its inode-stable commit lock behind, just like a
    canonical ``hs_<index>`` target.  The lock is evidence, not a cache entry;
    require the exact suffix, a zero-length regular file, and an immediately
    acquirable advisory lock so an active writer can never be sealed complete.
    """
    if (
        not lock_path.name.startswith(".")
        or not lock_path.name.endswith(".safetensors.commit.lock")
    ):
        raise ProvenanceError(
            f"Invalid hidden-state commit lock artifact: {lock_path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags)
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_size != 0:
            raise ProvenanceError(
                f"Invalid hidden-state commit lock artifact: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProvenanceError(
                f"Hidden-state commit lock is still active: {lock_path}"
            ) from error
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class ProvenanceError(ValueError):
    """Raised when an artifact cannot be safely created or reused."""


def validate_full_commit_sha(
    revision: str,
    field_name: str = "DATASET_REVISION",
) -> str:
    """Return a normalized immutable dataset revision or reject it."""
    if not _FULL_COMMIT_RE.fullmatch(revision):
        raise ProvenanceError(
            f"{field_name} must be an immutable full 40-character commit SHA; "
            f"got {revision!r}. Branches, tags, and abbreviated SHAs are unsafe."
        )
    return revision.lower()


def validate_local_model_snapshot(model_path: Path, model_revision: str) -> Path:
    """Return a canonical local HF snapshot whose basename is its commit SHA."""
    normalized_revision = validate_full_commit_sha(
        model_revision,
        "MODEL_REVISION",
    )
    candidate = model_path.expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"MODEL must be an existing local snapshot directory: {model_path}"
        ) from error
    if not resolved.is_dir():
        raise ProvenanceError(
            f"MODEL must be an existing local snapshot directory: {model_path}"
        )
    if resolved.name.lower() != normalized_revision:
        raise ProvenanceError(
            "MODEL must resolve to the exact local HF snapshot named by "
            f"MODEL_REVISION ({normalized_revision}); got {resolved}."
        )
    return resolved


def validate_qwen3_vl_4b_snapshot(model_path: Path) -> Path:
    """Validate the architecture-defining metadata of Qwen3-VL-4B-Instruct.

    A snapshot directory name proves which revision was requested, but it does
    not prove that the directory contains the model family advertised by the
    example.  Validate the stable shape metadata that distinguishes the 4B
    checkpoint from the other Qwen3-VL sizes before launching vLLM.
    """
    resolved = model_path.expanduser().resolve(strict=True)
    config_path = resolved / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(
            f"MODEL snapshot has no readable JSON config: {config_path}"
        ) from error
    if not isinstance(config, dict):
        raise ProvenanceError(f"MODEL config must contain a JSON object: {config_path}")

    text_config = config.get("text_config")
    vision_config = config.get("vision_config")
    expected_top_level = {
        "model_type": "qwen3_vl",
    }
    expected_text = {
        "model_type": "qwen3_vl_text",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
    }
    expected_vision = {
        "depth": 24,
        "hidden_size": 1024,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    mismatches: list[str] = []
    for key, expected in expected_top_level.items():
        if config.get(key) != expected:
            mismatches.append(
                f"config.{key}={config.get(key)!r} (expected {expected!r})"
            )
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or "Qwen3VLForConditionalGeneration" not in architectures
    ):
        mismatches.append(
            "config.architectures does not contain "
            "'Qwen3VLForConditionalGeneration'"
        )
    if not isinstance(text_config, dict):
        mismatches.append("config.text_config is not an object")
    else:
        for key, expected in expected_text.items():
            if text_config.get(key) != expected:
                mismatches.append(
                    f"config.text_config.{key}={text_config.get(key)!r} "
                    f"(expected {expected!r})"
                )
    if not isinstance(vision_config, dict):
        mismatches.append("config.vision_config is not an object")
    else:
        for key, expected in expected_vision.items():
            if vision_config.get(key) != expected:
                mismatches.append(
                    f"config.vision_config.{key}={vision_config.get(key)!r} "
                    f"(expected {expected!r})"
                )
    if mismatches:
        raise ProvenanceError(
            "MODEL is not the approved Qwen3-VL-4B-Instruct architecture: "
            + "; ".join(mismatches)
        )
    return resolved


def validate_source_checkout(source_dir: Path, source_revision: str) -> str:
    """Bind a claimed source revision to the clean Git checkout being executed."""
    normalized_revision = validate_full_commit_sha(
        source_revision,
        "SOURCE_REVISION",
    )
    try:
        resolved_source = source_dir.expanduser().resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"Source checkout does not exist: {source_dir}"
        ) from error
    if not resolved_source.is_dir():
        raise ProvenanceError(f"Source checkout is not a directory: {source_dir}")

    def run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(resolved_source), *arguments],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise ProvenanceError(
                "Git is required to bind SOURCE_REVISION to the executing checkout."
            ) from error

    inside = run_git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise ProvenanceError(
            "SOURCE_REVISION cannot be verified because the executing source "
            f"directory is not a Git worktree: {resolved_source}"
        )
    head = run_git("rev-parse", "--verify", "HEAD^{commit}")
    actual_revision = head.stdout.strip().lower()
    if head.returncode != 0 or not _FULL_COMMIT_RE.fullmatch(actual_revision):
        raise ProvenanceError(
            f"Could not resolve the source checkout HEAD: {resolved_source}"
        )
    if actual_revision != normalized_revision:
        raise ProvenanceError(
            "SOURCE_REVISION does not match the executing checkout: "
            f"expected {normalized_revision}, HEAD is {actual_revision}."
        )

    # Default example outputs live below the checkout and are intentionally
    # untracked. They do not change the code represented by HEAD, so only tracked
    # modifications invalidate the source claim (the server runner separately
    # freezes and verifies the complete source manifest).
    status = run_git("status", "--porcelain=v1", "--untracked-files=no")
    if status.returncode != 0:
        raise ProvenanceError(
            f"Could not verify source checkout cleanliness: {resolved_source}"
        )
    if status.stdout.strip():
        changed = status.stdout.splitlines()[:20]
        raise ProvenanceError(
            "Executing source checkout is not clean; SOURCE_REVISION would not "
            f"describe the code being run. Changed tracked entries: {changed}"
        )
    return normalized_revision


def parse_fields(raw_fields: list[str]) -> dict[str, str]:
    """Parse unique ``key=value`` provenance fields."""
    fields: dict[str, str] = {}
    for raw_field in raw_fields:
        key, separator, value = raw_field.partition("=")
        if not separator or not _FIELD_KEY_RE.fullmatch(key):
            raise ProvenanceError(
                "Provenance fields must use lowercase key=value syntax; "
                f"got {raw_field!r}."
            )
        if key in fields:
            raise ProvenanceError(f"Duplicate provenance field: {key}")
        if not value:
            raise ProvenanceError(f"Provenance field {key!r} must not be empty.")
        fields[key] = value
    if not fields:
        raise ProvenanceError("At least one provenance field is required.")
    return dict(sorted(fields.items()))


def fingerprint_fields(fields: dict[str, str]) -> str:
    """Return a stable SHA-256 fingerprint for an exact pipeline configuration."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fields": dict(sorted(fields.items())),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_core(
    artifact_dir: Path,
    fields: dict[str, str],
    guard_dirs: list[Path],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_dir": str(artifact_dir.resolve()),
        "fields": dict(sorted(fields.items())),
        "guard_dirs": sorted(str(path.resolve()) for path in guard_dirs),
    }


def _serialize_manifest(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_has_symlink_component(
    path: Path,
    *,
    trusted_root: Path | None = None,
) -> bool:
    """Reject symlinks in every existing lexical path component.

    ``Path.resolve()`` cannot be used for this check because it erases the very
    symlink boundary that provenance validation needs to detect.  Walk from the
    filesystem anchor instead, including ancestors of paths that do not exist
    yet.  When a trusted root is supplied, require lexical containment as well
    as checking the root and every child component.
    """
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    root: Path | None = None
    if trusted_root is not None:
        root = trusted_root.expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        try:
            candidate.relative_to(root)
        except ValueError:
            return True

    anchor = Path(candidate.anchor)
    current = anchor
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            return True

    if root is not None:
        current = Path(root.anchor)
        for component in root.parts[1:]:
            current /= component
            if current.is_symlink():
                return True
    return False


def _reject_temporary_artifact(relative_path: Path) -> None:
    name = relative_path.name.lower()
    if name.startswith(".nfs") or name.endswith(_TEMPORARY_FILE_SUFFIXES):
        raise ProvenanceError(
            f"Artifact contains a temporary file or directory: {relative_path}"
        )


def _hash_regular_file(file_path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(file_path, flags)
    except OSError as error:
        raise ProvenanceError(
            f"Could not safely open artifact file: {file_path}"
        ) from error

    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProvenanceError(f"Artifact entry is not a regular file: {file_path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ProvenanceError(f"Artifact file changed while hashing: {file_path}")
    return before.st_size, digest.hexdigest()


def _iter_artifact_files(root: Path) -> list[tuple[Path, Path]]:
    """Return ``(relative, absolute)`` files without following any symlink."""
    files: list[tuple[Path, Path]] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise ProvenanceError(
                f"Could not enumerate artifact directory: {directory}"
            ) from error
        for entry in entries:
            relative_path = relative_directory / entry.name
            _reject_temporary_artifact(relative_path)
            try:
                if entry.is_symlink():
                    raise ProvenanceError(
                        f"Artifact contains a symbolic link: {relative_path}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), relative_path)
                elif entry.is_file(follow_symlinks=False):
                    files.append((relative_path, Path(entry.path)))
                else:
                    raise ProvenanceError(
                        "Artifact contains a non-regular entry: "
                        f"{relative_path}"
                    )
            except OSError as error:
                raise ProvenanceError(
                    f"Could not inspect artifact entry: {relative_path}"
                ) from error

    visit(root, Path())
    return sorted(files, key=lambda item: item[0].as_posix())


def _content_aggregate(directory: Path) -> dict[str, Any]:
    """Hash every file as ``relative_path\0size\0sha256\0`` in sorted order."""
    _validate_completed_artifact(directory)
    resolved_root = directory.resolve(strict=True)
    aggregate = hashlib.sha256()
    file_count = 0
    total_size = 0
    for relative_path, file_path in _iter_artifact_files(resolved_root):
        size, file_digest = _hash_regular_file(file_path)
        record = (
            relative_path.as_posix().encode("utf-8", errors="surrogateescape")
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + file_digest.encode("ascii")
            + b"\0"
        )
        aggregate.update(record)
        file_count += 1
        total_size += size
    if file_count == 0:
        raise ProvenanceError(
            f"Completed artifact contains no regular files: {directory}"
        )
    return {
        "file_count": file_count,
        "sha256": aggregate.hexdigest(),
        "total_size": total_size,
    }


def _content_aggregates(directories: list[Path]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for directory in directories:
        aggregate = _content_aggregate(directory)
        aggregates[str(directory.resolve(strict=True))] = aggregate
    return aggregates


def _write_temp_payload(destination: Path, payload: dict[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        handle.write(_serialize_manifest(payload))
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _create_manifest_exclusively(
    destination: Path,
    payload: dict[str, Any],
) -> bool:
    temporary = _write_temp_payload(destination, payload)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return False
        _fsync_directory(destination.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _replace_manifest(destination: Path, payload: dict[str, Any]) -> None:
    temporary = _write_temp_payload(destination, payload)
    try:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_symlink():
        raise ProvenanceError(f"Manifest must not be a symlink: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(
            f"Could not read provenance manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ProvenanceError(
            f"Provenance manifest must contain a JSON object: {manifest_path}"
        )
    return payload


def _validate_manifest_payload(
    payload: dict[str, Any],
    artifact_dir: Path,
    fields: dict[str, str],
    guard_dirs: list[Path],
) -> str:
    state = payload.get("state")
    expected_core = {
        **_manifest_core(artifact_dir, fields, guard_dirs),
        "state": state,
    }
    if any(payload.get(key) != value for key, value in expected_core.items()):
        raise ProvenanceError(
            "Provenance mismatch. Refusing to reuse artifacts because the manifest "
            "does not exactly match this run's configuration and artifact path."
        )
    if state not in {"in_progress", "complete"}:
        raise ProvenanceError(f"Unknown provenance manifest state: {state!r}")
    if state == "in_progress":
        if payload != expected_core:
            raise ProvenanceError(
                "An in-progress provenance manifest contains unexpected fields."
            )
        return state

    artifact_directories = _artifact_directories(artifact_dir, guard_dirs)
    actual_aggregates = _content_aggregates(artifact_directories)
    expected_complete = {
        **expected_core,
        "content_aggregates": actual_aggregates,
    }
    if payload != expected_complete:
        raise ProvenanceError(
            "Artifact content does not match its completed provenance manifest. "
            "Refusing stale, modified, incomplete, or mixed reuse."
        )
    return state


def _artifact_directories(
    artifact_dir: Path,
    guard_dirs: list[Path],
) -> list[Path]:
    directories: dict[str, Path] = {}
    for directory in [artifact_dir, *guard_dirs]:
        directories[str(directory.resolve())] = directory
    return list(directories.values())


def _validate_manifest_location(
    manifest_path: Path,
    artifact_directories: list[Path],
) -> None:
    if _path_has_symlink_component(manifest_path):
        raise ProvenanceError(
            f"Manifest path contains a symbolic-link component: {manifest_path}"
        )
    resolved_manifest = manifest_path.resolve(strict=False)
    for directory in artifact_directories:
        resolved_directory = directory.resolve(strict=False)
        try:
            resolved_manifest.relative_to(resolved_directory)
        except ValueError:
            continue
        raise ProvenanceError(
            "The provenance manifest must be outside every hashed artifact "
            f"directory to avoid self-reference: {manifest_path}"
        )


def _validate_fresh_artifact(directory: Path) -> None:
    if _path_has_symlink_component(directory):
        raise ProvenanceError(
            f"Artifact directory contains a symbolic-link component: {directory}"
        )
    if directory.exists():
        if not directory.is_dir():
            raise ProvenanceError(
                f"Artifact path exists but is not a directory: {directory}"
            )
        if any(directory.iterdir()):
            raise ProvenanceError(
                "Found existing artifacts without a provenance manifest at "
                f"{directory}. Refusing to mix them with this run."
            )


def _validate_completed_artifact(directory: Path) -> None:
    if _path_has_symlink_component(directory):
        raise ProvenanceError(
            "Completed artifact directory contains a symbolic-link component: "
            f"{directory}"
        )
    if not directory.is_dir() or not any(directory.iterdir()):
        raise ProvenanceError(
            "Completed provenance manifest points to a missing or empty artifact: "
            f"{directory}"
        )


def claim_artifact(
    manifest_path: Path,
    artifact_dir: Path,
    fields: dict[str, str],
    guard_dirs: list[Path] | None = None,
) -> str:
    """Atomically claim a fresh artifact, or verify a completed exact match.

    Returns ``"run"`` for a newly claimed artifact and ``"reuse"`` for an
    already completed artifact. An interrupted or concurrent claim fails closed.
    """
    guard_dirs = guard_dirs or []
    artifact_directories = _artifact_directories(artifact_dir, guard_dirs)
    _validate_manifest_location(manifest_path, artifact_directories)
    if manifest_path.exists() or manifest_path.is_symlink():
        state = _validate_manifest_payload(
            _load_manifest(manifest_path), artifact_dir, fields, guard_dirs
        )
        if state == "complete":
            for directory in artifact_directories:
                _validate_completed_artifact(directory)
            return "reuse"
        raise ProvenanceError(
            f"Artifact has an unfinished provenance claim: {manifest_path}. "
            "Use a new ATTEMPT_LABEL or inspect and remove the incomplete "
            "attempt explicitly."
        )

    for directory in artifact_directories:
        _validate_fresh_artifact(directory)

    payload = {
        **_manifest_core(artifact_dir, fields, guard_dirs),
        "state": "in_progress",
    }
    if _create_manifest_exclusively(manifest_path, payload):
        return "run"

    state = _validate_manifest_payload(
        _load_manifest(manifest_path), artifact_dir, fields, guard_dirs
    )
    if state == "complete":
        for directory in artifact_directories:
            _validate_completed_artifact(directory)
        return "reuse"
    raise ProvenanceError(
        f"Another process already claimed this artifact: {manifest_path}"
    )


def complete_artifact(
    manifest_path: Path,
    artifact_dir: Path,
    fields: dict[str, str],
    guard_dirs: list[Path] | None = None,
) -> None:
    """Atomically mark a successfully produced artifact as complete."""
    guard_dirs = guard_dirs or []
    artifact_directories = _artifact_directories(artifact_dir, guard_dirs)
    _validate_manifest_location(manifest_path, artifact_directories)
    if not manifest_path.exists() or manifest_path.is_symlink():
        raise ProvenanceError(
            f"Cannot complete an artifact without a claim: {manifest_path}"
        )
    state = _validate_manifest_payload(
        _load_manifest(manifest_path), artifact_dir, fields, guard_dirs
    )
    if state == "complete":
        for directory in artifact_directories:
            _validate_completed_artifact(directory)
        return
    for directory in artifact_directories:
        _validate_completed_artifact(directory)
    payload = {
        **_manifest_core(artifact_dir, fields, guard_dirs),
        "state": "complete",
        "content_aggregates": _content_aggregates(artifact_directories),
    }
    _replace_manifest(manifest_path, payload)


def _as_list(value: object, *, field: str, row_index: int) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ProvenanceError(
            f"Prepared row {row_index} field {field} must be a list."
        )
    return value


def _extract_prepared_image_url(part: dict[str, Any], *, row_index: int) -> str:
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if not isinstance(image_url, dict) or set(image_url) != {"url"}:
            raise ProvenanceError(
                f"Prepared row {row_index} has a malformed image_url content part."
            )
        reference = image_url.get("url")
    elif part_type in {"image", "input_image"}:
        candidates = [
            part.get(key)
            for key in ("image", "path", "url")
            if part.get(key) is not None
        ]
        if len(candidates) != 1:
            raise ProvenanceError(
                f"Prepared row {row_index} has an ambiguous image content part."
            )
        reference = candidates[0]
    else:  # pragma: no cover - caller constrains the type
        raise AssertionError(f"unexpected image part type: {part_type}")
    if not isinstance(reference, str) or not reference:
        raise ProvenanceError(
            f"Prepared row {row_index} image reference must be a non-empty string."
        )
    return reference


def _validate_prepared_image_reference(
    reference: str,
    *,
    dataset_dir: Path,
    row_index: int,
) -> None:
    parsed = urlsplit(reference)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise ProvenanceError(
            f"Prepared row {row_index} image must use an unambiguous local file URI."
        )
    decoded_path = unquote(parsed.path)
    if not decoded_path or "\x00" in decoded_path:
        raise ProvenanceError(
            f"Prepared row {row_index} image URI contains an invalid local path."
        )
    candidate = Path(decoded_path)
    if not candidate.is_absolute():
        raise ProvenanceError(
            f"Prepared row {row_index} image URI path must be absolute."
        )
    try:
        resolved_image = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"Prepared row {row_index} image file does not exist: {candidate}"
        ) from error
    if not resolved_image.is_file():
        raise ProvenanceError(
            f"Prepared row {row_index} image is not a regular file: {candidate}"
        )
    try:
        resolved_image.relative_to(dataset_dir)
    except ValueError as error:
        raise ProvenanceError(
            f"Prepared row {row_index} image escapes dataset_dir: {resolved_image}"
        ) from error
    if _path_has_symlink_component(candidate, trusted_root=dataset_dir):
        raise ProvenanceError(
            f"Prepared row {row_index} image path contains a symbolic link."
        )


def _validate_prepared_messages(
    messages: object,
    *,
    dataset_dir: Path,
    row_index: int,
) -> None:
    if not isinstance(messages, list) or not messages:
        raise ProvenanceError(
            f"Prepared row {row_index} messages must be a non-empty list."
        )

    image_references: list[str] = []
    user_images = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ProvenanceError(
                f"Prepared row {row_index} message {message_index} is not an object."
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ProvenanceError(
                f"Prepared row {row_index} message {message_index} has no role."
            )
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ProvenanceError(
                f"Prepared row {row_index} message {message_index} has invalid content."
            )
        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                raise ProvenanceError(
                    f"Prepared row {row_index} contains a non-object content part."
                )
            part_type = part.get("type")
            if part_type in _UNSUPPORTED_MEDIA_PART_TYPES:
                raise ProvenanceError(
                    f"Prepared row {row_index} contains unsupported media {part_type}."
                )
            if part_type in _IMAGE_PART_TYPES:
                image_references.append(
                    _extract_prepared_image_url(part, row_index=row_index)
                )
                if role == "user":
                    user_images += 1

    if len(image_references) != 1 or user_images != 1:
        raise ProvenanceError(
            f"Prepared row {row_index} must contain exactly one user image; "
            f"found {len(image_references)} total image(s), "
            f"{user_images} in user turns."
        )
    _validate_prepared_image_reference(
        image_references[0],
        dataset_dir=dataset_dir,
        row_index=row_index,
    )


def validate_prepared_dataset(
    prepared_dir: Path,
    *,
    dataset_dir: Path,
    max_samples: int,
    seq_length: int,
) -> int:
    """Load prepared Arrow data and validate its row and sequence-size contract."""
    if max_samples <= 0 or seq_length <= 0:
        raise ProvenanceError("max_samples and seq_length must both be positive.")
    if _path_has_symlink_component(prepared_dir) or not prepared_dir.is_dir():
        raise ProvenanceError(
            f"Prepared dataset must be a real directory: {prepared_dir}"
        )
    if _path_has_symlink_component(dataset_dir) or not dataset_dir.is_dir():
        raise ProvenanceError(
            f"Dataset directory must be a real directory: {dataset_dir}"
        )
    dataset_dir = dataset_dir.resolve(strict=True)
    prepared_dir = prepared_dir.resolve(strict=True)

    from datasets import load_from_disk  # noqa: PLC0415

    dataset = load_from_disk(str(prepared_dir))
    row_count = len(dataset)
    if row_count != max_samples:
        raise ProvenanceError(
            "Prepared dataset row count does not match the approved exact count "
            f"{max_samples}: {row_count}"
        )
    required_columns = {"input_ids", "loss_mask", "messages", "seq_len"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ProvenanceError(
            f"Prepared dataset is missing columns: {sorted(missing_columns)}"
        )

    for index, row in enumerate(dataset):
        input_ids = _as_list(row["input_ids"], field="input_ids", row_index=index)
        loss_mask = _as_list(row["loss_mask"], field="loss_mask", row_index=index)
        input_length = len(input_ids)
        if input_length <= 0 or input_length > seq_length:
            raise ProvenanceError(
                f"Prepared row {index} has invalid sequence length "
                f"{input_length} (approved maximum {seq_length})."
            )
        if len(loss_mask) != len(input_ids):
            raise ProvenanceError(
                f"Prepared row {index} has mismatched input_ids/loss_mask lengths."
            )
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, numbers.Integral)
            for token_id in input_ids
        ):
            raise ProvenanceError(
                f"Prepared row {index} input_ids must contain only integers."
            )
        if any(
            not isinstance(mask_value, (bool, numbers.Integral))
            or int(mask_value) not in {0, 1}
            for mask_value in loss_mask
        ):
            raise ProvenanceError(
                f"Prepared row {index} loss_mask must contain only binary values."
            )
        if not any(int(mask_value) == 1 for mask_value in loss_mask):
            raise ProvenanceError(
                f"Prepared row {index} loss_mask has no trainable token."
            )
        row_seq_len = row["seq_len"]
        if (
            isinstance(row_seq_len, bool)
            or not isinstance(row_seq_len, numbers.Integral)
            or int(row_seq_len) != input_length
        ):
            raise ProvenanceError(
                f"Prepared row {index} seq_len does not equal len(input_ids)."
            )
        _validate_prepared_messages(
            row["messages"],
            dataset_dir=dataset_dir,
            row_index=index,
        )
    return row_count


def _load_json_object(file_path: Path, *, label: str) -> dict[str, Any]:
    if file_path.is_symlink() or not file_path.is_file():
        raise ProvenanceError(f"{label} must be a regular JSON file: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"Could not parse {label}: {file_path}") from error
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{label} must contain a JSON object: {file_path}")
    return payload


def _validate_model_safetensors(file_path: Path) -> None:
    if file_path.is_symlink() or not file_path.is_file():
        raise ProvenanceError(
            f"Checkpoint model weights must be a regular file: {file_path}"
        )
    from safetensors import SafetensorError, safe_open  # noqa: PLC0415

    try:
        with safe_open(str(file_path), framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if not keys:
                raise ProvenanceError(
                    f"Checkpoint model safetensors contains no tensors: {file_path}"
                )
            for key in keys:
                tensor_slice = tensors.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                if any(dimension <= 0 for dimension in shape):
                    raise ProvenanceError(
                        "Checkpoint model contains an empty tensor "
                        f"{key!r} with shape {shape}."
                    )
                # Read the checkpoint in bounded first-axis chunks so a NaN/Inf
                # cannot be sealed into a completed artifact without loading a
                # large embedding or projection tensor all at once.
                if not shape:
                    chunks = [tensors.get_tensor(key)]
                else:
                    chunks = (
                        tensor_slice[start : start + 128]
                        for start in range(0, shape[0], 128)
                    )
                for chunk in chunks:
                    if chunk.is_floating_point() and not bool(
                        chunk.isfinite().all().item()
                    ):
                        raise ProvenanceError(
                            "Checkpoint model contains NaN or Inf values in "
                            f"tensor {key!r}."
                        )
    except SafetensorError as error:
        raise ProvenanceError(
            f"Invalid checkpoint model safetensors: {file_path}"
        ) from error


def _validate_optimizer_checkpoint(file_path: Path) -> None:
    """Require a loadable, finite optimizer payload rather than non-empty bytes."""
    if file_path.is_symlink() or not file_path.is_file():
        raise ProvenanceError(
            f"Checkpoint optimizer state must be a regular file: {file_path}"
        )
    import torch  # noqa: PLC0415

    try:
        payload = torch.load(
            file_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise ProvenanceError(
            f"Checkpoint optimizer state is not loadable: {file_path}"
        ) from error
    if not isinstance(payload, (dict, list, tuple)) or not payload:
        raise ProvenanceError(
            f"Checkpoint optimizer state has an invalid root object: {file_path}"
        )

    stack: list[object] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(value.isfinite().all().item()):
                raise ProvenanceError(
                    "Checkpoint optimizer state contains NaN or Inf values: "
                    f"{file_path}"
                )


def _validate_runtime_hidden_state(
    file_path: Path,
    *,
    hidden_states_dir: Path,
    expected_tokens: list[int],
) -> None:
    from safetensors import SafetensorError, safe_open  # noqa: PLC0415

    from speculators.data_generation.offline import (  # noqa: PLC0415
        InvalidHiddenStateCacheError,
        validate_hidden_states_file_contents,
    )

    try:
        validate_hidden_states_file_contents(
            file_path,
            hidden_states_dir,
            expected_tokens=expected_tokens,
            validate_values=True,
        )
        with safe_open(str(file_path), framework="pt", device="cpu") as tensors:
            hidden_shape = tuple(tensors.get_slice("hidden_states").get_shape())
    except (InvalidHiddenStateCacheError, SafetensorError, OSError) as error:
        raise ProvenanceError(
            f"Invalid runtime hidden-state cache entry: {file_path}"
        ) from error
    if len(hidden_shape) != 3 or hidden_shape[1] < 2:
        raise ProvenanceError(
            "Runtime hidden_states must have shape "
            f"[seq_len, at_least_two_layers, hidden_size], got {hidden_shape} "
            f"in {file_path}."
        )


def _validate_checkpoint_directory(
    checkpoint_dir: Path,
    *,
    epochs: int,
) -> None:
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ProvenanceError(
            f"Checkpoint directory must be a real directory: {checkpoint_dir}"
        )
    numeric_directories: dict[int, Path] = {}
    for entry in checkpoint_dir.iterdir():
        if entry.is_symlink():
            raise ProvenanceError(f"Checkpoint directory contains a symlink: {entry}")
        if entry.is_dir() and entry.name.isdigit():
            epoch = int(entry.name)
            if entry.name != str(epoch):
                raise ProvenanceError(
                    f"Checkpoint epoch directory is not canonical: {entry}"
                )
            numeric_directories[epoch] = entry
            continue
        if entry.is_file() and entry.name == "train_command.txt":
            continue
        raise ProvenanceError(f"Unexpected checkpoint-root artifact: {entry}")

    expected_epochs = set(range(epochs))
    if set(numeric_directories) != expected_epochs:
        raise ProvenanceError(
            "Checkpoint epoch set is incomplete or mixed: expected "
            f"{sorted(expected_epochs)}, got {sorted(numeric_directories)}."
        )

    for epoch, epoch_dir in sorted(numeric_directories.items()):
        _load_json_object(epoch_dir / "config.json", label="checkpoint config")
        _validate_model_safetensors(epoch_dir / "model.safetensors")
        optimizer_path = epoch_dir / "optimizer_state_dict.pt"
        _validate_optimizer_checkpoint(optimizer_path)
        training_state = _load_json_object(
            epoch_dir / "training_state.json",
            label="checkpoint training state",
        )
        saved_epoch = training_state.get("epoch")
        if (
            isinstance(saved_epoch, bool)
            or not isinstance(saved_epoch, numbers.Integral)
            or int(saved_epoch) != epoch
        ):
            raise ProvenanceError(
                f"Checkpoint {epoch} training_state has the wrong epoch."
            )
        local_step = training_state.get("local_step")
        if (
            isinstance(local_step, bool)
            or not isinstance(local_step, numbers.Integral)
            or int(local_step) != 0
        ):
            raise ProvenanceError(
                f"Checkpoint {epoch} is a mid-epoch rather than completed checkpoint."
            )
        global_step = training_state.get("global_step")
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, numbers.Integral)
            or int(global_step) <= 0
        ):
            raise ProvenanceError(
                f"Checkpoint {epoch} has an invalid completed global_step."
            )


def validate_runtime_artifacts(
    checkpoint_dir: Path,
    hidden_states_dir: Path,
    prepared_dir: Path,
    *,
    max_samples: int,
    epochs: int,
) -> int:
    """Validate exact 5k cache ownership and completed numeric checkpoints."""
    if max_samples <= 0 or epochs <= 0:
        raise ProvenanceError("max_samples and epochs must both be positive.")
    if hidden_states_dir.is_symlink() or not hidden_states_dir.is_dir():
        raise ProvenanceError(
            f"Hidden-state directory must be a real directory: {hidden_states_dir}"
        )
    if prepared_dir.is_symlink() or not prepared_dir.is_dir():
        raise ProvenanceError(
            f"Prepared dataset must be a real directory: {prepared_dir}"
        )

    from datasets import load_from_disk  # noqa: PLC0415

    prepared = load_from_disk(str(prepared_dir.resolve(strict=True)))
    if len(prepared) != max_samples:
        raise ProvenanceError(
            "Runtime validation expected the prepared dataset to contain exactly "
            f"{max_samples} rows, got {len(prepared)}."
        )

    expected_names = {f"hs_{index}.safetensors" for index in range(max_samples)}
    actual_safetensors: set[str] = set()
    for entry in hidden_states_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ProvenanceError(
                f"Hidden-state directory contains a non-regular entry: {entry}"
            )
        if entry.suffix == ".safetensors":
            actual_safetensors.add(entry.name)
        elif entry.name.startswith(".") and entry.name.endswith(
            ".safetensors.commit.lock"
        ):
            _validate_inactive_commit_lock(entry)
        else:
            raise ProvenanceError(
                f"Hidden-state directory contains an unexpected artifact: {entry}"
            )
    if actual_safetensors != expected_names:
        missing = sorted(expected_names - actual_safetensors)
        extra = sorted(actual_safetensors - expected_names)
        raise ProvenanceError(
            "Runtime hidden-state cache set is not exact: "
            f"missing_count={len(missing)}, missing_sample={missing[:20]}, "
            f"extra_count={len(extra)}, extra_sample={extra[:20]}."
        )

    resolved_hidden_states = hidden_states_dir.resolve(strict=True)
    for index in range(max_samples):
        input_ids = prepared[index]["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if not isinstance(input_ids, list):
            raise ProvenanceError(
                f"Prepared row {index} input_ids is not a list during "
                "runtime validation."
            )
        if any(
            isinstance(token, bool) or not isinstance(token, numbers.Integral)
            for token in input_ids
        ):
            raise ProvenanceError(
                f"Prepared row {index} input_ids contains a non-integer token."
            )
        expected_tokens = [int(token) for token in input_ids]
        _validate_runtime_hidden_state(
            resolved_hidden_states / f"hs_{index}.safetensors",
            hidden_states_dir=resolved_hidden_states,
            expected_tokens=expected_tokens,
        )

    _validate_checkpoint_directory(checkpoint_dir, epochs=epochs)
    return max_samples


def validate_train_parquet_shards(dataset_dir: Path) -> list[Path]:
    """Reject incomplete, stale, mixed, or symlinked train Parquet shards."""
    data_dir = dataset_dir / "data"
    if dataset_dir.is_symlink() or data_dir.is_symlink():
        raise ProvenanceError("Dataset and data directories must not be symlinks.")
    if not data_dir.is_dir():
        raise ProvenanceError(f"Dataset data directory does not exist: {data_dir}")

    entries = sorted(data_dir.iterdir())
    if not entries:
        raise ProvenanceError(f"No Parquet shards found in {data_dir}")

    parsed: list[tuple[Path, int, int]] = []
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ProvenanceError(f"Unexpected non-file dataset entry: {entry}")
        match = _TRAIN_SHARD_RE.fullmatch(entry.name)
        if match is None:
            raise ProvenanceError(
                "Unexpected dataset file; refusing possible stale shard mix: "
                f"{entry}"
            )
        parsed.append((entry, int(match.group(1)), int(match.group(2))))

    totals = {total for _, _, total in parsed}
    if len(totals) != 1:
        raise ProvenanceError(
            f"Parquet shards declare mixed totals in {data_dir}: {sorted(totals)}"
        )
    total = totals.pop()
    indices = [index for _, index, _ in parsed]
    expected_indices = list(range(total))
    if sorted(indices) != expected_indices:
        raise ProvenanceError(
            "Parquet shard indices are incomplete or duplicated: "
            f"expected {expected_indices}, got {sorted(indices)}"
        )
    if len(parsed) != total:
        raise ProvenanceError(
            f"Expected {total} Parquet shards, found {len(parsed)} in {data_dir}"
        )
    return [path for path, _, _ in sorted(parsed, key=lambda item: item[1])]


def _add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--field", action="append", default=[], dest="fields")
    parser.add_argument(
        "--guard-dir", action="append", default=[], type=Path, dest="guard_dirs"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    revision_parser = subparsers.add_parser("validate-revision")
    revision_parser.add_argument("revision")
    revision_parser.add_argument(
        "--field-name",
        choices=("DATASET_REVISION", "SOURCE_REVISION", "MODEL_REVISION"),
        default="DATASET_REVISION",
    )

    model_parser = subparsers.add_parser("validate-model-snapshot")
    model_parser.add_argument("--model", type=Path, required=True)
    model_parser.add_argument("--revision", required=True)
    model_parser.add_argument(
        "--expected-profile",
        choices=("qwen3-vl-4b-instruct",),
    )

    source_parser = subparsers.add_parser("validate-source-checkout")
    source_parser.add_argument("--source-dir", type=Path, required=True)
    source_parser.add_argument("--revision", required=True)

    fingerprint_parser = subparsers.add_parser("fingerprint")
    fingerprint_parser.add_argument(
        "--field", action="append", default=[], dest="fields"
    )

    claim_parser = subparsers.add_parser("claim")
    _add_manifest_arguments(claim_parser)

    complete_parser = subparsers.add_parser("complete")
    _add_manifest_arguments(complete_parser)

    shards_parser = subparsers.add_parser("validate-shards")
    shards_parser.add_argument("--dataset-dir", type=Path, required=True)

    prepared_parser = subparsers.add_parser("validate-prepared")
    prepared_parser.add_argument("--prepared-dir", type=Path, required=True)
    prepared_parser.add_argument("--dataset-dir", type=Path, required=True)
    prepared_parser.add_argument("--max-samples", type=int, required=True)
    prepared_parser.add_argument("--seq-length", type=int, required=True)

    runtime_parser = subparsers.add_parser("validate-runtime")
    runtime_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    runtime_parser.add_argument("--hidden-states-dir", type=Path, required=True)
    runtime_parser.add_argument("--prepared-dir", type=Path, required=True)
    runtime_parser.add_argument("--max-samples", type=int, required=True)
    runtime_parser.add_argument("--epochs", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-revision":
            print(validate_full_commit_sha(args.revision, args.field_name))
        elif args.command == "validate-model-snapshot":
            model_path = validate_local_model_snapshot(args.model, args.revision)
            if args.expected_profile == "qwen3-vl-4b-instruct":
                model_path = validate_qwen3_vl_4b_snapshot(model_path)
            print(model_path)
        elif args.command == "validate-source-checkout":
            print(validate_source_checkout(args.source_dir, args.revision))
        elif args.command == "fingerprint":
            print(fingerprint_fields(parse_fields(args.fields)))
        elif args.command == "claim":
            print(
                claim_artifact(
                    args.manifest,
                    args.artifact_dir,
                    parse_fields(args.fields),
                    args.guard_dirs,
                )
            )
        elif args.command == "complete":
            complete_artifact(
                args.manifest,
                args.artifact_dir,
                parse_fields(args.fields),
                args.guard_dirs,
            )
        elif args.command == "validate-shards":
            shards = validate_train_parquet_shards(args.dataset_dir)
            print(f"validated {len(shards)} train Parquet shard(s)")
        elif args.command == "validate-prepared":
            row_count = validate_prepared_dataset(
                args.prepared_dir,
                dataset_dir=args.dataset_dir,
                max_samples=args.max_samples,
                seq_length=args.seq_length,
            )
            print(f"validated {row_count} prepared row(s)")
        elif args.command == "validate-runtime":
            validated = validate_runtime_artifacts(
                args.checkpoint_dir,
                args.hidden_states_dir,
                args.prepared_dir,
                max_samples=args.max_samples,
                epochs=args.epochs,
            )
            print(
                f"validated {validated} runtime hidden-state file(s) and "
                f"{args.epochs} checkpoint epoch(s)"
            )
        else:  # pragma: no cover - argparse constrains this value
            parser.error(f"Unknown command: {args.command}")
    except ProvenanceError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
