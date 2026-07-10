import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_HIDDEN_STATE_NAME = re.compile(r"hs_(\d+)\.safetensors")
_INVALID_DIRECTORY_NAME = "invalid"


class InvalidHiddenStateCacheError(ValueError):
    """A deterministic cache defect that is safe to quarantine for regeneration."""


def _save_safetensors(tensors: dict, file_path: Path) -> None:
    """Import the torch adapter only when serialization is actually requested."""
    from safetensors.torch import save_file  # noqa: PLC0415

    save_file(tensors, file_path)


def _path_has_symlink_component(path: Path) -> bool:
    """Return whether any existing component in an absolute path is a symlink."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def validate_hidden_states_root(
    allowed_root: str | Path,
    *,
    require_exists: bool = True,
) -> Path:
    """Validate and resolve an explicitly configured hidden-state root."""
    root = Path(allowed_root).expanduser()
    if not root.is_absolute():
        root = root.absolute()
    if _path_has_symlink_component(root):
        raise ValueError(
            f"Allowed hidden-state root contains a symlink: {allowed_root}"
        )
    if not root.exists():
        if not require_exists:
            return root.resolve(strict=False)
        raise ValueError(
            f"Allowed hidden-state root must be an existing directory: {allowed_root}"
        )
    if not root.is_dir():
        raise ValueError(
            f"Allowed hidden-state root must be an existing directory: {allowed_root}"
        )
    return root.resolve(strict=True)


def validate_hidden_states_path(
    file_path: str | Path,
    allowed_root: str | Path,
    *,
    require_exists: bool = True,
) -> Path:
    """Validate an endpoint-provided hidden-state path before local file access.

    The response path must be absolute, have a ``.safetensors`` suffix, contain no
    symlink component, and resolve beneath the explicitly allowed root. Set
    ``require_exists=False`` before waiting on a connector lock; the parent must
    still exist, and callers must run the strict default validation after waiting.
    """
    candidate = Path(file_path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"Hidden-state path must be absolute: {file_path}")
    if candidate.suffix != ".safetensors":
        raise ValueError(f"Hidden-state path must end in .safetensors: {file_path}")
    if _path_has_symlink_component(candidate):
        raise ValueError(f"Hidden-state path contains a symlink component: {file_path}")

    resolved_root = validate_hidden_states_root(allowed_root)
    if not candidate.parent.exists() or not candidate.parent.is_dir():
        raise ValueError(
            f"Hidden-state parent must be an existing directory: {candidate.parent}"
        )
    if require_exists and not candidate.exists():
        raise ValueError(f"Hidden-state file does not exist: {file_path}")
    resolved_candidate = candidate.resolve(strict=require_exists)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as e:
        raise ValueError(
            "Hidden-state path is outside the allowed root: "
            f"{resolved_candidate} not under {resolved_root}"
        ) from e

    if not candidate.exists():
        return resolved_candidate
    if not candidate.is_file():
        raise ValueError(f"Hidden-state path is not a regular file: {file_path}")

    return resolved_candidate


def validate_hidden_states_file_contents(
    file_path: str | Path,
    allowed_root: str | Path,
    *,
    expected_tokens: Sequence[int] | None = None,
    validate_values: bool = False,
) -> list[int]:
    """Validate the safetensors structure used by offline training.

    Only the token tensor is materialized. The hidden-state tensor is inspected
    through its safetensors slice metadata so resume scanning does not load every
    cached hidden-state tensor into memory.
    """
    try:
        validated_path = validate_hidden_states_path(file_path, allowed_root)
    except ValueError as e:
        raise InvalidHiddenStateCacheError(str(e)) from e

    # Keep torch/safetensors lazy: path-only callers of this module must not need
    # the torch adapter to be importable.
    from safetensors import SafetensorError, safe_open  # noqa: PLC0415

    try:
        with safe_open(str(validated_path), framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            missing = {"token_ids", "hidden_states"} - keys
            if missing:
                raise InvalidHiddenStateCacheError(
                    "Hidden-state safetensors is missing required tensors: "
                    + ", ".join(sorted(missing))
                )

            token_ids = tensors.get_tensor("token_ids")
            hidden_shape = tuple(
                tensors.get_slice("hidden_states").get_shape()
            )
    except SafetensorError as e:
        raise InvalidHiddenStateCacheError(
            f"Invalid hidden-state safetensors payload: {e}"
        ) from e

    if token_ids.ndim != 1:
        raise InvalidHiddenStateCacheError(
            "Hidden-state token_ids must be one-dimensional, "
            f"got shape {tuple(token_ids.shape)}"
        )
    if token_ids.numel() == 0:
        raise InvalidHiddenStateCacheError("Hidden-state token_ids must not be empty")
    dtype_name = str(token_ids.dtype)
    if dtype_name not in {"torch.int32", "torch.int64"}:
        raise InvalidHiddenStateCacheError(
            "Hidden-state token_ids must use a training-compatible integer dtype "
            f"(torch.int32 or torch.int64), got {token_ids.dtype}"
        )
    if len(hidden_shape) != 3 or any(dimension <= 0 for dimension in hidden_shape):
        raise InvalidHiddenStateCacheError(
            "Hidden-state tensor must have exactly three non-empty dimensions "
            "[seq_len, num_layers, hidden_size], "
            f"got shape {hidden_shape}"
        )
    if hidden_shape[0] != token_ids.shape[0]:
        raise InvalidHiddenStateCacheError(
            "Hidden-state sequence length does not match token_ids: "
            f"{hidden_shape[0]} != {token_ids.shape[0]}"
        )

    # Read a single element to validate the framework dtype without loading the
    # full cache tensor. Resume validation optionally streams the first axis so
    # NaN/Inf cannot turn a corrupt cache into a false completed sample.
    try:
        with safe_open(str(validated_path), framework="pt", device="cpu") as tensors:
            hidden_slice = tensors.get_slice("hidden_states")
            hidden_probe = hidden_slice[0:1, 0:1, 0:1]
            if not hidden_probe.is_floating_point():
                raise InvalidHiddenStateCacheError(
                    "Hidden-state tensor must use a floating dtype, "
                    f"got {hidden_probe.dtype}"
                )
            if validate_values:
                chunk_size = 128
                for start in range(0, hidden_shape[0], chunk_size):
                    chunk = hidden_slice[start : start + chunk_size, :, :]
                    if not bool(chunk.isfinite().all().item()):
                        raise InvalidHiddenStateCacheError(
                            "Hidden-state tensor contains NaN or Inf values"
                        )
    except SafetensorError as e:
        raise InvalidHiddenStateCacheError(
            f"Invalid hidden-state safetensors payload: {e}"
        ) from e

    actual_tokens = [int(token) for token in token_ids.tolist()]
    if expected_tokens is not None:
        if hasattr(expected_tokens, "tolist"):
            expected_tokens = expected_tokens.tolist()
        normalized_expected = [int(token) for token in expected_tokens]
        if actual_tokens != normalized_expected:
            raise InvalidHiddenStateCacheError(
                "Hidden-state token_ids do not match current dataset input_ids"
            )

    return actual_tokens


def _temporary_path_for(target_path: Path) -> Path:
    fd, temp_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    return Path(temp_name)


def _fsync_file(file_path: Path) -> None:
    with file_path.open("rb") as file_obj:
        os.fsync(file_obj.fileno())


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directories(*directories: Path) -> None:
    for directory in dict.fromkeys(path.resolve() for path in directories):
        _fsync_directory(directory)


def _target_lock_path(target_path: Path, allowed_root: str | Path) -> Path:
    """Return a safe, persistent cooperative-lock path for a target."""
    root = validate_hidden_states_root(allowed_root)
    target = Path(target_path).expanduser()
    if not target.is_absolute():
        target = target.absolute()
    if target.suffix != ".safetensors":
        raise ValueError(f"Hidden-state path must end in .safetensors: {target_path}")
    if _path_has_symlink_component(target.parent):
        raise ValueError(
            f"Hidden-state parent contains a symlink component: {target.parent}"
        )
    if not target.parent.exists() or not target.parent.is_dir():
        raise ValueError(
            f"Hidden-state parent must be an existing directory: {target.parent}"
        )
    resolved_parent = target.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"Hidden-state target is outside the allowed root: {target_path}"
        ) from e
    normalized_target = resolved_parent / target.name
    return normalized_target.with_name(f".{normalized_target.name}.commit.lock")


@contextmanager
def _target_commit_lock(
    target_path: Path,
    allowed_root: str | Path,
) -> Iterator[None]:
    """Serialize cooperative commits without following a lock-file symlink."""
    lock_path = _target_lock_path(target_path, allowed_root)
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ValueError(
                f"Hidden-state commit lock is a symlink: {lock_path}"
            ) from e
        raise

    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(
                f"Hidden-state commit lock is not a regular file: {lock_path}"
            )
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _files_are_byte_identical(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as first_file, second.open("rb") as second_file:
        while True:
            first_chunk = first_file.read(1024 * 1024)
            second_chunk = second_file.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hidden_states_file_sha256(
    file_path: str | Path,
    *,
    allowed_root: str | Path,
) -> str:
    """Return a streaming digest after revalidating a managed cache file."""
    validated_path = validate_hidden_states_path(file_path, allowed_root)
    return _sha256_file(validated_path)


def _existing_target_is_idempotent(
    staged_path: Path,
    target_path: Path,
    *,
    allowed_root: str | Path,
    allow_replace: bool,
    expected_existing_sha256: str | None = None,
) -> bool:
    """Return True for an identical target, otherwise reject unsafe clobbering."""
    if not os.path.lexists(target_path):
        return False
    existing_target = validate_hidden_states_path(target_path, allowed_root)
    if _files_are_byte_identical(staged_path, existing_target):
        return True
    if not allow_replace:
        raise FileExistsError(
            errno.EEXIST,
            "Refusing to replace an existing hidden-state target with different "
            "bytes",
            str(existing_target),
        )
    if expected_existing_sha256 is None:
        raise ValueError(
            "allow_replace requires the SHA256 of the exact source version that "
            "was validated before the in-place rewrite"
        )
    actual_existing_sha256 = hidden_states_file_sha256(
        existing_target,
        allowed_root=allowed_root,
    )
    if actual_existing_sha256 != expected_existing_sha256:
        raise FileExistsError(
            errno.EEXIST,
            "Refusing an in-place hidden-state rewrite because the target changed "
            "after it was read",
            str(existing_target),
        )
    return False


def durable_unlink_safetensors(
    file_path: str | Path,
    *,
    allowed_root: str | Path,
    expected_sha256: str | None = None,
) -> None:
    """Remove exactly the validated source version and durably commit the removal.

    The source is first atomically claimed under a private name. This closes the
    validate-then-unlink race: if another writer replaces the pathname, the
    replacement is checked (and restored) instead of being silently deleted.
    """
    validated_path = validate_hidden_states_path(file_path, allowed_root)
    parent = validated_path.parent

    with _target_commit_lock(validated_path, allowed_root):
        validated_path = validate_hidden_states_path(validated_path, allowed_root)
        claimed_path = parent / (
            f".{validated_path.name}.{uuid.uuid4().hex}.delete.safetensors"
        )
        os.replace(validated_path, claimed_path)

        try:
            claimed_path = validate_hidden_states_path(claimed_path, allowed_root)
            if expected_sha256 is not None:
                claimed_sha256 = hidden_states_file_sha256(
                    claimed_path,
                    allowed_root=allowed_root,
                )
                if claimed_sha256 != expected_sha256:
                    raise FileExistsError(
                        errno.EEXIST,
                        "Refusing to delete a hidden-state source because it changed "
                        "after it was validated",
                        str(validated_path),
                    )
            claimed_path.unlink()
            _fsync_directory(parent)
        except BaseException:
            # Restore the atomically claimed entry whenever no newer writer has
            # recreated the public source pathname. If it has, preserve both
            # versions and surface the private evidence path in the exception.
            if os.path.lexists(claimed_path) and not os.path.lexists(validated_path):
                os.replace(claimed_path, validated_path)
                _fsync_directory(parent)
            raise


def atomic_save_safetensors(
    tensors: dict,
    target_path: str | Path,
    *,
    allowed_root: str | Path,
    allow_replace: bool = False,
    expected_existing_sha256: str | None = None,
) -> Path:
    """Serialize and atomically commit without silently clobbering a target.

    ``allow_replace`` is reserved for an explicitly validated in-place
    prefix-truncation rewrite. All ordinary writes reject a different existing
    target; a byte-identical target is treated as an idempotent success.
    """
    target = validate_hidden_states_path(
        target_path,
        allowed_root,
        require_exists=False,
    )
    temp_path = _temporary_path_for(target)
    try:
        _save_safetensors(tensors, temp_path)
        _fsync_file(temp_path)
        with _target_commit_lock(target, allowed_root):
            target = validate_hidden_states_path(
                target,
                allowed_root,
                require_exists=False,
            )
            if _existing_target_is_idempotent(
                temp_path,
                target,
                allowed_root=allowed_root,
                allow_replace=allow_replace,
                expected_existing_sha256=expected_existing_sha256,
            ):
                return validate_hidden_states_path(target, allowed_root)
            os.replace(temp_path, target)
            _fsync_directory(target.parent)
            return validate_hidden_states_path(target, allowed_root)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_move_safetensors(
    source_path: str | Path,
    target_path: str | Path,
    *,
    source_root: str | Path,
    target_root: str | Path,
    expected_source_sha256: str | None = None,
    expected_tokens: Sequence[int] | None = None,
) -> Path:
    """Atomically move one validated source version into a managed target.

    Direct renames validate the committed target before returning and roll back
    an unexpected source replacement. Cross-device moves copy through a private
    target-side file, bind the copied bytes to a digest, and delete the source
    only when the same digest is still present.
    """
    source = validate_hidden_states_path(source_path, source_root)
    target = validate_hidden_states_path(
        target_path,
        target_root,
        require_exists=False,
    )
    if source == target:
        return source

    source_parent = source.parent
    target_parent = target.parent
    cross_device = False
    with _target_commit_lock(target, target_root):
        source = validate_hidden_states_path(source, source_root)
        target = validate_hidden_states_path(
            target,
            target_root,
            require_exists=False,
        )
        source_sha256_for_delete = expected_source_sha256
        if os.path.lexists(target):
            if expected_tokens is not None:
                validate_hidden_states_file_contents(
                    source,
                    source_root,
                    expected_tokens=expected_tokens,
                )
            if source_sha256_for_delete is None:
                source_sha256_for_delete = hidden_states_file_sha256(
                    source,
                    allowed_root=source_root,
                )
            elif (
                hidden_states_file_sha256(source, allowed_root=source_root)
                != source_sha256_for_delete
            ):
                raise FileExistsError(
                    errno.EEXIST,
                    "Refusing to move a hidden-state source because it changed "
                    "after it was validated",
                    str(source),
                )
        if _existing_target_is_idempotent(
            source,
            target,
            allowed_root=target_root,
            allow_replace=False,
        ):
            durable_unlink_safetensors(
                source,
                allowed_root=source_root,
                expected_sha256=source_sha256_for_delete,
            )
            return validate_hidden_states_path(target, target_root)
        try:
            os.replace(source, target)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            cross_device = True
        else:
            try:
                target = validate_hidden_states_path(target, target_root)
                if expected_source_sha256 is not None and (
                    hidden_states_file_sha256(target, allowed_root=target_root)
                    != expected_source_sha256
                ):
                    raise FileExistsError(
                        errno.EEXIST,
                        "Hidden-state source changed during the atomic move",
                        str(target),
                    )
                if expected_tokens is not None:
                    validate_hidden_states_file_contents(
                        target,
                        target_root,
                        expected_tokens=expected_tokens,
                    )
            except BaseException:
                # The direct rename consumed the source pathname. Put the
                # unexpected version back so a concurrent writer's data is not
                # left behind under the canonical target name.
                if os.path.lexists(target) and not os.path.lexists(source):
                    os.replace(target, source)
                    _fsync_directories(target_parent, source_parent)
                raise
            _fsync_directories(target_parent, source_parent)
            return validate_hidden_states_path(target, target_root)

    if not cross_device:
        raise AssertionError("unreachable hidden-state move state")

    temp_path = _temporary_path_for(target)
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, flags)
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise ValueError(
                    f"Hidden-state source is not a regular file: {source}"
                )
            with os.fdopen(source_fd, "rb", closefd=False) as source_file:
                with temp_path.open("wb") as temp_file:
                    shutil.copyfileobj(source_file, temp_file, 1024 * 1024)
        finally:
            os.close(source_fd)
        _fsync_file(temp_path)
        copied_sha256 = _sha256_file(temp_path)
        if (
            expected_source_sha256 is not None
            and copied_sha256 != expected_source_sha256
        ):
            raise FileExistsError(
                errno.EEXIST,
                "Hidden-state source changed while it was copied across devices",
                str(source),
            )
        source_sha256_for_delete = expected_source_sha256 or copied_sha256

        with _target_commit_lock(target, target_root):
            source = validate_hidden_states_path(source, source_root)
            if (
                hidden_states_file_sha256(source, allowed_root=source_root)
                != source_sha256_for_delete
            ):
                raise FileExistsError(
                    errno.EEXIST,
                    "Refusing a cross-device move because the hidden-state source "
                    "changed after it was copied",
                    str(source),
                )
            if expected_tokens is not None:
                validate_hidden_states_file_contents(
                    source,
                    source_root,
                    expected_tokens=expected_tokens,
                )
            target = validate_hidden_states_path(
                target,
                target_root,
                require_exists=False,
            )
            if _existing_target_is_idempotent(
                temp_path,
                target,
                allowed_root=target_root,
                allow_replace=False,
            ):
                durable_unlink_safetensors(
                    source,
                    allowed_root=source_root,
                    expected_sha256=source_sha256_for_delete,
                )
                return validate_hidden_states_path(target, target_root)
            os.replace(temp_path, target)
            committed_target = True
            try:
                _fsync_directory(target.parent)
                target = validate_hidden_states_path(target, target_root)
                if _sha256_file(target) != copied_sha256:
                    raise FileExistsError(
                        errno.EEXIST,
                        "Cross-device hidden-state target changed during commit",
                        str(target),
                    )
                if expected_tokens is not None:
                    validate_hidden_states_file_contents(
                        target,
                        target_root,
                        expected_tokens=expected_tokens,
                    )
                durable_unlink_safetensors(
                    source,
                    allowed_root=source_root,
                    expected_sha256=source_sha256_for_delete,
                )
                committed_target = False
                return target
            finally:
                if committed_target and os.path.lexists(target):
                    rollback_target = validate_hidden_states_path(target, target_root)
                    # Remove only the exact copy installed by this operation.
                    # A non-cooperative replacement must never be deleted as part
                    # of rollback.
                    if _sha256_file(rollback_target) == copied_sha256:
                        rollback_target.unlink()
                        _fsync_directory(rollback_target.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def validate_generated_source_ownership(
    source_path: str | Path,
    target_path: str | Path,
    *,
    source_root: str | Path,
    target_root: str | Path,
    allow_current_target: bool = False,
) -> None:
    """Prevent a connector response from consuming another managed cache entry."""
    resolved_source_root = validate_hidden_states_root(source_root)
    resolved_target_root = validate_hidden_states_root(target_root)
    source = validate_hidden_states_path(source_path, resolved_source_root)
    target = validate_hidden_states_path(
        target_path,
        resolved_target_root,
        require_exists=False,
    )
    # Root equality is insufficient: a broad source root may contain the target
    # root. Treat every canonical sibling of the requested target as managed,
    # regardless of how the two configured roots overlap.
    if source.parent != target.parent:
        return
    if _HIDDEN_STATE_NAME.fullmatch(source.name) is None:
        return
    if source != target:
        raise ValueError(
            "Generated hidden-state source points to another managed cache entry: "
            f"{source.name} (expected staging file for {target.name})"
        )
    if not allow_current_target:
        raise ValueError(
            "Generated hidden-state source may equal the current managed target "
            "only for an explicit in-place prefix truncation"
        )


def validate_hidden_states_tensors(data: dict) -> None:
    """Validate the in-memory training schema for one hidden-state sample."""
    missing = {"token_ids", "hidden_states"} - set(data)
    if missing:
        raise ValueError(
            "Hidden-state data is missing required tensors: "
            + ", ".join(sorted(missing))
        )

    token_ids = data["token_ids"]
    if token_ids.ndim != 1:
        raise ValueError(
            "Hidden-state token_ids must be one-dimensional, "
            f"got shape {tuple(token_ids.shape)}"
        )
    if token_ids.numel() == 0:
        raise ValueError("Hidden-state token_ids must not be empty")
    if str(token_ids.dtype) not in {"torch.int32", "torch.int64"}:
        raise ValueError(
            "Hidden-state token_ids must use torch.int32 or torch.int64, "
            f"got {token_ids.dtype}"
        )

    hs = data["hidden_states"]
    if hs.ndim != 3 or any(dimension <= 0 for dimension in hs.shape):
        raise ValueError(
            "Hidden states must have exactly three non-empty dimensions "
            "[seq_len, num_layers, hidden_size]; "
            f"got {tuple(hs.shape)}"
        )
    if not hs.is_floating_point():
        raise ValueError(f"Hidden states must use a floating dtype, got {hs.dtype}")
    if not hs.isfinite().all():
        raise ValueError("Hidden states contain NaN or Inf values")
    if token_ids.shape[0] != hs.shape[0]:
        raise ValueError(
            f"Sequence length of hidden states {hs.shape[0]}"
            f" doesn't match num tokens {token_ids.shape[0]}"
        )


def check_hidden_states(data: dict, tokens: list[int]):
    validate_hidden_states_tensors(data)
    t_ids = [int(token) for token in data["token_ids"].tolist()]
    normalized_tokens = [int(token) for token in tokens]
    if t_ids != normalized_tokens:
        raise ValueError(
            f"Token ids don't match expected token ids {normalized_tokens}"
        )


def align_hidden_states_to_tokens(
    data: dict,
    tokens: list[int],
    *,
    allow_prefix_truncation: bool = False,
) -> tuple[dict, bool]:
    """Validate and optionally trim hidden states to the preprocessed token prefix."""
    validate_hidden_states_tensors(data)
    t_ids = [int(token) for token in data["token_ids"].tolist()]
    tokens = [int(token) for token in tokens]
    if t_ids == tokens:
        check_hidden_states(data, tokens)
        return data, False

    expected_len = len(tokens)
    if (
        allow_prefix_truncation
        and t_ids[:expected_len] == tokens
        and data["hidden_states"].shape[0] >= expected_len
    ):
        aligned = dict(data)
        aligned["token_ids"] = data["token_ids"][:expected_len].contiguous()
        aligned["hidden_states"] = data["hidden_states"][:expected_len].contiguous()
        check_hidden_states(aligned, tokens)
        return aligned, True

    check_hidden_states(data, tokens)
    return data, False


def _reason_slug(reason: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-")
    return slug[:64] or "invalid"


def _write_quarantine_record(
    record_path: Path,
    *,
    original_path: Path,
    quarantined_path: Path,
    reason: str,
    timestamp: datetime,
) -> None:
    record = {
        "timestamp": timestamp.isoformat(),
        "original_path": str(original_path),
        "quarantined_path": str(quarantined_path),
        "reason": reason,
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(record_path, flags, 0o600)
    try:
        payload = (json.dumps(record, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _quarantine_invalid_hidden_state(
    file_path: Path,
    output_root: Path,
    *,
    reason: str,
) -> Path | None:
    """Move an invalid cache entry aside while preserving evidence and metadata."""
    if not os.path.lexists(file_path):
        return None

    invalid_dir = output_root / _INVALID_DIRECTORY_NAME
    created_invalid_dir = False
    try:
        invalid_dir.mkdir(mode=0o700)
        created_invalid_dir = True
    except FileExistsError:
        pass
    if invalid_dir.is_symlink() or not invalid_dir.is_dir():
        raise ValueError(
            f"Hidden-state quarantine path is not a safe directory: {invalid_dir}"
        )
    if created_invalid_dir:
        _fsync_directory(output_root)

    timestamp = datetime.now(timezone.utc)
    timestamp_label = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    reason_label = _reason_slug(reason)
    unique = uuid.uuid4().hex[:12]
    quarantined_path = invalid_dir / (
        f"{file_path.stem}.invalid-{reason_label}-{timestamp_label}-{unique}"
        f"{file_path.suffix}"
    )
    record_path = quarantined_path.with_name(quarantined_path.name + ".json")

    os.replace(file_path, quarantined_path)
    _fsync_directories(output_root, invalid_dir)
    _write_quarantine_record(
        record_path,
        original_path=file_path,
        quarantined_path=quarantined_path,
        reason=reason,
        timestamp=timestamp,
    )
    _fsync_directory(invalid_dir)
    logger.warning(
        "Quarantined invalid hidden-state cache entry %s as %s: %s",
        file_path,
        quarantined_path,
        reason,
    )
    return quarantined_path


def get_existing_hidden_state_indices(
    output_path: Path,
    *,
    expected_tokens_for_index: Callable[[int], Sequence[int]] | None = None,
) -> list[int]:
    """Return only valid resumable ``hs_<index>.safetensors`` entries.

    Invalid entries are moved to ``<output>/invalid`` with a UTC timestamp,
    reason-bearing filename, and a durable JSON evidence record. This leaves the
    canonical target free for regeneration instead of silently skipping it.
    """
    if not output_path.exists():
        return []
    output_root = validate_hidden_states_root(output_path)
    existing_file_indices_set: set[int] = set()

    for file_path in sorted(output_root.iterdir()):
        match = _HIDDEN_STATE_NAME.fullmatch(file_path.name)
        if match is None:
            continue
        file_index = int(match.group(1))

        with _target_commit_lock(file_path, output_root):
            if not os.path.lexists(file_path):
                continue
            entry_stat = file_path.lstat()
            invalid_error: InvalidHiddenStateCacheError | None = None
            if file_path.name != f"hs_{file_index}.safetensors":
                invalid_error = InvalidHiddenStateCacheError(
                    "cache entry does not use the canonical index filename"
                )
            elif stat.S_ISLNK(entry_stat.st_mode):
                invalid_error = InvalidHiddenStateCacheError(
                    "cache entry is a symbolic link"
                )
            elif not stat.S_ISREG(entry_stat.st_mode):
                invalid_error = InvalidHiddenStateCacheError(
                    "cache entry is not a regular file"
                )

            expected_tokens = None
            if invalid_error is None and expected_tokens_for_index is not None:
                try:
                    expected_tokens = expected_tokens_for_index(file_index)
                except InvalidHiddenStateCacheError as e:
                    invalid_error = e

            if invalid_error is None:
                try:
                    validate_hidden_states_file_contents(
                        file_path,
                        output_root,
                        expected_tokens=expected_tokens,
                        validate_values=True,
                    )
                except InvalidHiddenStateCacheError as e:
                    invalid_error = e

            if invalid_error is not None:
                _quarantine_invalid_hidden_state(
                    file_path,
                    output_root,
                    reason=str(invalid_error),
                )
                continue

            existing_file_indices_set.add(file_index)

    return sorted(existing_file_indices_set)


def get_indices_to_process(
    num_samples: int,
    max_samples: int | None,
    existing: list[int],
    world_size: int,
    rank: int,
) -> list[int]:
    """Determines which indices should be processed. If max_samples is None
    returns all dataset indices not in existing. Otherwise gets the first
    `max_samples - len(existing)` samples not already in existing.

    Args:
        num_samples: Total size of preprocessed dataset
        max_samples: (Optional) limit for number of samples to process
        existing: list of ids that have already been processed
        world_size: Number of nodes to generate on
        rank: The rank of the local node

    Returns:
        list of dataset indices to process
    """

    target = min(max_samples, num_samples) if max_samples is not None else num_samples

    if target <= 0:
        return []

    chunk_size = target // world_size
    remainder = target % world_size
    # Distribute remainder across the first `remainder` ranks so chunks differ
    # by at most 1.
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)

    existing_s = set(existing)
    to_process = [i for i in range(start, end) if i not in existing_s]

    if not to_process:
        logger.info("All samples for this rank already processed!")
        return []

    if len(existing_s & set(range(start, end))) > 0:
        logger.info(
            f"Found {len(existing_s & set(range(start, end)))} existing samples"
            f" for rank {rank}."
        )

    return to_process
