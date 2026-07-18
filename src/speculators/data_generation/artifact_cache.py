from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, save_file

from speculators.data_generation.vllm_client import ClientItem

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

_REQUEST_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STATS_VERSION = 1
_COUNTERS = (
    "logical_requests",
    "hits",
    "misses",
    "coalesced_waiters",
    "retry_generations",
    "publishes",
    "generation_failures",
    "publish_failures",
    "invalid_artifacts_removed",
    "expired_artifacts_removed",
    "stale_temps_removed",
    "lock_timeouts",
)


def canonical_hidden_state_extraction_namespace(
    target_layer_ids: list[int] | tuple[int, ...],
    *,
    user_namespace: str | None = None,
) -> str:
    """Fingerprint the producer configuration that changes artifact semantics."""

    if (
        not target_layer_ids
        or not all(
            isinstance(layer_id, int) and not isinstance(layer_id, bool)
            for layer_id in target_layer_ids
        )
        or len(set(target_layer_ids)) != len(target_layer_ids)
    ):
        raise ValueError("target_layer_ids must be non-empty unique integers")
    if user_namespace is not None and not user_namespace:
        raise ValueError("user_namespace must be non-empty when provided")
    identity = {
        "schema_version": 1,
        "target_layer_ids": list(target_layer_ids),
    }
    if user_namespace is not None:
        identity["user_namespace"] = user_namespace
    return json.dumps(
        identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


class ArtifactCacheError(RuntimeError):
    """Base error raised by shared hidden-state artifact caching."""


class ArtifactLockTimeoutError(ArtifactCacheError, TimeoutError):
    """Raised when a request key remains owned beyond the configured timeout."""


@dataclass(frozen=True)
class ArtifactResult:
    data: dict[str, torch.Tensor]
    request_id: str
    path: Path
    cache_hit: bool
    coalesced: bool


def canonical_hidden_state_request_id(
    model: str,
    client_item: ClientItem,
    *,
    namespace: str | None = None,
) -> str:
    """Build a stable identity for the hidden states produced by one request."""
    if not model:
        raise ValueError("model must be non-empty")
    token_ids = client_item.get("input_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in token_ids
        )
    ):
        raise ValueError("input_ids must be one non-empty list of integer token IDs")

    identity: dict[str, Any] = {
        "input_ids": token_ids,
        "model": model,
        "schema_version": 1,
    }
    messages = client_item.get("messages")
    if messages is not None:
        identity["messages"] = messages
    if namespace is not None:
        identity["namespace"] = namespace
    try:
        encoded = json.dumps(
            identity, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("request identity must be JSON serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _empty_stats() -> dict[str, int]:
    return {"schema_version": _STATS_VERSION, **dict.fromkeys(_COUNTERS, 0)}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class HiddenStateArtifactCache:
    """Cross-process, publish-once cache for immutable hidden-state tensors."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        artifact_ttl_seconds: float | None = 3600.0,
        stale_temp_seconds: float = 300.0,
        lock_timeout_seconds: float = 300.0,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        if artifact_ttl_seconds is not None and artifact_ttl_seconds <= 0:
            raise ValueError("artifact_ttl_seconds must be positive or None")
        if stale_temp_seconds <= 0:
            raise ValueError("stale_temp_seconds must be positive")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        if lock_poll_seconds <= 0:
            raise ValueError("lock_poll_seconds must be positive")

        self.root = Path(root).expanduser().resolve()
        self.artifact_ttl_seconds = artifact_ttl_seconds
        self.stale_temp_seconds = stale_temp_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_poll_seconds = lock_poll_seconds
        self._artifacts = self.root / "artifacts"
        self._locks = self.root / "locks"
        self._stats_path = self.root / "stats.json"
        self._stats_lock_path = self.root / "stats.lock"
        self._artifacts.mkdir(parents=True, exist_ok=True)
        self._locks.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
            raise ValueError("request_id must be a lowercase SHA-256 digest")

    def artifact_path(self, request_id: str) -> Path:
        self._validate_request_id(request_id)
        return self._artifacts / request_id[:2] / f"{request_id}.safetensors"

    def _lock_path(self, request_id: str) -> Path:
        return self._locks / request_id[:2] / f"{request_id}.lock"

    @contextmanager
    def _request_lock(
        self, request_id: str, *, timeout_seconds: float | None = None
    ) -> Iterator[bool]:
        lock_path = self._lock_path(request_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        timeout = (
            self.lock_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        waited = False
        acquired = False
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    waited = True
                    remaining = deadline - time.monotonic()
                    if timeout == 0 or remaining <= 0:
                        raise ArtifactLockTimeoutError(
                            f"Timed out waiting for hidden-state request {request_id}"
                        ) from None
                    time.sleep(min(self.lock_poll_seconds, remaining))
            yield waited
        finally:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_stats_unlocked(self) -> dict[str, int]:
        if not self._stats_path.exists():
            return _empty_stats()
        try:
            value = json.loads(self._stats_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactCacheError("Cache accounting file is unreadable") from error
        if not isinstance(value, dict) or value.get("schema_version") != _STATS_VERSION:
            raise ArtifactCacheError("Cache accounting schema is invalid")
        stats = _empty_stats()
        for counter in _COUNTERS:
            count = value.get(counter)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ArtifactCacheError(
                    f"Cache accounting counter {counter!r} is invalid"
                )
            stats[counter] = count
        return stats

    @contextmanager
    def _stats_lock(self, operation: int) -> Iterator[None]:
        descriptor = os.open(self._stats_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, operation)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _record(self, **deltas: int) -> None:
        unknown = set(deltas) - set(_COUNTERS)
        if unknown:
            raise ValueError(f"Unknown cache accounting counters: {sorted(unknown)}")
        if any(delta < 0 for delta in deltas.values()):
            raise ValueError("Cache accounting deltas must be non-negative")
        with self._stats_lock(fcntl.LOCK_EX):
            stats = self._read_stats_unlocked()
            for counter, delta in deltas.items():
                stats[counter] += delta
            temporary = self.root / f".stats.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("w") as output:
                    json.dump(stats, output, separators=(",", ":"), sort_keys=True)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(self._stats_path)
                _fsync_directory(self.root)
            finally:
                temporary.unlink(missing_ok=True)

    def snapshot_stats(self) -> dict[str, int]:
        with self._stats_lock(fcntl.LOCK_SH):
            return self._read_stats_unlocked()

    def record_reuse(self) -> None:
        """Account for a logical reader served by an existing publication."""
        self._record(logical_requests=1, hits=1)

    def load(
        self,
        request_id: str,
        validate: Callable[[dict[str, torch.Tensor]], None],
    ) -> dict[str, torch.Tensor]:
        """Load a publication while holding its cross-process file lock."""
        self._validate_request_id(request_id)
        with self._request_lock(request_id):
            path = self.artifact_path(request_id)
            if not path.exists():
                raise ArtifactCacheError(f"Artifact {request_id} is not published")
            data = load_file(path)
            validate(data)
            return data

    def remove(self, request_id: str, *, expected_path: Path | None = None) -> bool:
        """Remove one publication under the same lock used by readers/writers."""
        self._validate_request_id(request_id)
        with self._request_lock(request_id):
            path = self.artifact_path(request_id)
            if expected_path is not None and path.resolve() != expected_path.resolve():
                raise ArtifactCacheError(
                    f"Artifact path mismatch for request {request_id}"
                )
            if not path.exists():
                return False
            path.unlink()
            _fsync_directory(path.parent)
            return True

    def _is_expired(self, path: Path, now: float) -> bool:
        return bool(
            self.artifact_ttl_seconds is not None
            and now - path.stat().st_mtime >= self.artifact_ttl_seconds
        )

    def _remove_stale_temps_for_key(self, request_id: str, now: float) -> int:
        artifact_path = self.artifact_path(request_id)
        removed = 0
        for path in artifact_path.parent.glob(f".{request_id}.*.tmp"):
            try:
                if now - path.stat().st_mtime >= self.stale_temp_seconds:
                    path.unlink(missing_ok=True)
                    removed += 1
            except FileNotFoundError:
                pass
        return removed

    @staticmethod
    def _validate_tensors(data: Mapping[str, torch.Tensor]) -> None:
        if not data or not all(
            isinstance(value, torch.Tensor) for value in data.values()
        ):
            raise ArtifactCacheError("Artifact producer must return a tensor mapping")

    def _publish(
        self, request_id: str, data: Mapping[str, torch.Tensor], target: Path
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / (
            f".{request_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            save_file(dict(data), temporary)
            with temporary.open("rb") as published:
                os.fsync(published.fileno())
            temporary.replace(target)
            _fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def get_or_create(
        self,
        request_id: str,
        create: Callable[[], dict[str, torch.Tensor]],
        validate: Callable[[dict[str, torch.Tensor]], None],
    ) -> ArtifactResult:
        """Load a published artifact or elect one caller to create it."""
        self._validate_request_id(request_id)
        try:
            lock = self._request_lock(request_id)
            with lock as waited:
                now = time.time()
                stale_temps = self._remove_stale_temps_for_key(request_id, now)
                artifact_path = self.artifact_path(request_id)
                expired = 0
                invalid = 0
                if artifact_path.exists() and self._is_expired(artifact_path, now):
                    artifact_path.unlink(missing_ok=True)
                    expired = 1

                if artifact_path.exists():
                    try:
                        data = load_file(artifact_path)
                        validate(data)
                    except Exception:
                        artifact_path.unlink(missing_ok=True)
                        invalid = 1
                    else:
                        self._record(
                            logical_requests=1,
                            hits=1,
                            coalesced_waiters=int(waited),
                            stale_temps_removed=stale_temps,
                            expired_artifacts_removed=expired,
                        )
                        return ArtifactResult(
                            data=data,
                            request_id=request_id,
                            path=artifact_path,
                            cache_hit=True,
                            coalesced=waited,
                        )

                try:
                    data = create()
                    self._validate_tensors(data)
                    validate(data)
                except BaseException:
                    self._record(
                        logical_requests=1,
                        misses=1,
                        generation_failures=1,
                        retry_generations=int(waited),
                        invalid_artifacts_removed=invalid,
                        expired_artifacts_removed=expired,
                        stale_temps_removed=stale_temps,
                    )
                    raise
                try:
                    self._publish(request_id, data, artifact_path)
                except BaseException:
                    self._record(
                        logical_requests=1,
                        misses=1,
                        publish_failures=1,
                        retry_generations=int(waited),
                        invalid_artifacts_removed=invalid,
                        expired_artifacts_removed=expired,
                        stale_temps_removed=stale_temps,
                    )
                    raise

                self._record(
                    logical_requests=1,
                    misses=1,
                    publishes=1,
                    retry_generations=int(waited),
                    invalid_artifacts_removed=invalid,
                    expired_artifacts_removed=expired,
                    stale_temps_removed=stale_temps,
                )
                return ArtifactResult(
                    data=data,
                    request_id=request_id,
                    path=artifact_path,
                    cache_hit=False,
                    coalesced=False,
                )
        except ArtifactLockTimeoutError:
            self._record(logical_requests=1, lock_timeouts=1)
            raise

    def cleanup_stale(self, *, now: float | None = None) -> dict[str, int]:
        """Remove expired artifacts and abandoned temporary cache writes."""
        current_time = time.time() if now is None else now
        expired = 0
        stale_temps = 0

        for path in self._artifacts.glob("*/*.safetensors"):
            request_id = path.stem
            if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
                continue
            try:
                with self._request_lock(request_id, timeout_seconds=0):
                    if path.exists() and self._is_expired(path, current_time):
                        path.unlink(missing_ok=True)
                        expired += 1
            except ArtifactLockTimeoutError:
                continue

        for path in self._artifacts.glob("*/.*.tmp"):
            name = path.name
            request_id = name[1:65]
            if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
                continue
            try:
                with self._request_lock(request_id, timeout_seconds=0):
                    if (
                        path.exists()
                        and current_time - path.stat().st_mtime
                        >= self.stale_temp_seconds
                    ):
                        path.unlink(missing_ok=True)
                        stale_temps += 1
            except ArtifactLockTimeoutError:
                continue

        with self._stats_lock(fcntl.LOCK_EX):
            for path in self.root.glob(".stats.*.tmp"):
                try:
                    if (
                        path.exists()
                        and current_time - path.stat().st_mtime
                        >= self.stale_temp_seconds
                    ):
                        path.unlink(missing_ok=True)
                        stale_temps += 1
                except FileNotFoundError:
                    pass

        if expired or stale_temps:
            self._record(
                expired_artifacts_removed=expired,
                stale_temps_removed=stale_temps,
            )
        return {
            "expired_artifacts_removed": expired,
            "stale_temps_removed": stale_temps,
        }
