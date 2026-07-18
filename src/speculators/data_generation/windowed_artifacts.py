"""Bounded asynchronous coordination for shared hidden-state artifacts.

The coordinator is a single-host control plane. Tensor payloads remain in an
artifact store; SQLite contains only deterministic stream positions, consumer
progress, generation claims, and read leases.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 2
DIGEST_LENGTH = 64
MAX_CONSUMER_ID_LENGTH = 128


class WindowedArtifactError(RuntimeError):
    """Base error for bounded artifact coordination."""


class ArtifactGenerationError(WindowedArtifactError):
    """A producer exhausted the configured generation attempts."""


class ArtifactState(str, Enum):
    ABSENT = "absent"
    QUEUED = "queued"
    GENERATING = "generating"
    READY = "ready"
    EVICTING = "evicting"
    FAILED = "failed"


class ArtifactPriority(IntEnum):
    DEMAND = 0
    PREFETCH = 1


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_stream_id(contract: Mapping[str, Any]) -> str:
    """Return a stable identity for a dataset and sampler-order contract."""
    return hashlib.sha256(_canonical_json(contract).encode()).hexdigest()


def canonical_position_id(
    stream_id: str,
    *,
    epoch: int,
    ordinal: int,
    dataset_index: int,
    batch_ordinal: int,
    batch_start_sequence: int,
    batch_end_sequence: int,
) -> str:
    value = {
        "batch_end_sequence": batch_end_sequence,
        "batch_ordinal": batch_ordinal,
        "batch_start_sequence": batch_start_sequence,
        "dataset_index": dataset_index,
        "epoch": epoch,
        "ordinal": ordinal,
        "stream_id": stream_id,
    }
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class StreamSampleIndex:
    """Dataset index annotated with its deterministic sampler position."""

    stream_id: str
    sequence: int
    epoch: int
    ordinal: int
    dataset_index: int
    batch_ordinal: int
    batch_start_sequence: int
    batch_end_sequence: int
    request_id: str
    position_id: str

    def __post_init__(self) -> None:
        for name in ("stream_id", "request_id", "position_id"):
            value = getattr(self, name)
            if len(value) != DIGEST_LENGTH or any(
                ch not in "0123456789abcdef" for ch in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "sequence",
            "epoch",
            "ordinal",
            "dataset_index",
            "batch_ordinal",
            "batch_start_sequence",
            "batch_end_sequence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.batch_start_sequence <= self.sequence < self.batch_end_sequence:
            raise ValueError("sequence must be inside its batch boundaries")


@dataclass(frozen=True)
class ArtifactReadLease:
    token: str
    consumer_id: str
    stream_id: str
    sequence: int
    request_id: str
    path: Path
    generation: int
    cache_hit: bool
    wait_seconds: float

    def as_batch_metadata(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "consumer_id": self.consumer_id,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "request_id": self.request_id,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class GenerationClaim:
    request_id: str
    stream_id: str
    dataset_index: int
    generation: int
    priority: ArtifactPriority


@dataclass(frozen=True)
class EvictionClaim:
    request_id: str
    generation: int
    path: Path


class WindowedArtifactCoordinator:
    """Transactional authority for independent consumer windows."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        poll_seconds: float = 0.02,
        consumer_timeout_seconds: float = 120.0,
        claim_timeout_seconds: float = 300.0,
        max_generation_attempts: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if consumer_timeout_seconds <= 0:
            raise ValueError("consumer_timeout_seconds must be positive")
        if claim_timeout_seconds <= 0:
            raise ValueError("claim_timeout_seconds must be positive")
        if max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be at least one")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "windowed-artifacts.sqlite3"
        self.poll_seconds = poll_seconds
        self.consumer_timeout_seconds = consumer_timeout_seconds
        self.claim_timeout_seconds = claim_timeout_seconds
        self.max_generation_attempts = max_generation_attempts
        self._clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        if not self._schema_is_current():
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_schema()

    def _schema_is_current(self) -> bool:
        table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='coordinator_meta'"
        ).fetchone()
        if table is None:
            return False
        row = self._conn.execute(
            "SELECT value FROM coordinator_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            return False
        observed = int(row["value"])
        if observed != SCHEMA_VERSION:
            raise WindowedArtifactError(
                f"unsupported coordinator schema version {observed}; "
                f"expected {SCHEMA_VERSION}"
            )
        return True

    def _create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS coordinator_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS streams (
            stream_id TEXT PRIMARY KEY,
            contract_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
            stream_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            dataset_index INTEGER NOT NULL,
            batch_ordinal INTEGER NOT NULL,
            batch_start_sequence INTEGER NOT NULL,
            batch_end_sequence INTEGER NOT NULL,
            request_id TEXT NOT NULL,
            position_id TEXT NOT NULL UNIQUE,
            PRIMARY KEY (stream_id, sequence),
            UNIQUE (stream_id, epoch, ordinal),
            FOREIGN KEY (stream_id) REFERENCES streams(stream_id)
        );
        CREATE INDEX IF NOT EXISTS positions_request
            ON positions(request_id, stream_id);
        CREATE TABLE IF NOT EXISTS consumers (
            consumer_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            cursor INTEGER NOT NULL,
            lookbehind INTEGER NOT NULL,
            lookahead INTEGER NOT NULL,
            max_prefetch INTEGER NOT NULL,
            max_inflight INTEGER NOT NULL,
            state TEXT NOT NULL,
            heartbeat_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (stream_id) REFERENCES streams(stream_id)
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            request_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0,
            path TEXT,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            priority INTEGER,
            queued_at REAL,
            claim_owner TEXT,
            claim_until REAL,
            failures INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            first_reader_accounted INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS artifacts_schedule
            ON artifacts(state, priority, queued_at);
        CREATE TABLE IF NOT EXISTS interests (
            consumer_id TEXT NOT NULL,
            stream_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            request_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('window', 'demand')),
            created_at REAL NOT NULL,
            PRIMARY KEY (consumer_id, stream_id, sequence, kind),
            FOREIGN KEY (consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS interests_request ON interests(request_id);
        CREATE TABLE IF NOT EXISTS acquisitions (
            token TEXT PRIMARY KEY,
            consumer_id TEXT NOT NULL,
            stream_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            request_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('waiting', 'leased')),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS acquisitions_request
            ON acquisitions(request_id);
        CREATE TABLE IF NOT EXISTS completed_positions (
            consumer_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            PRIMARY KEY (consumer_id, sequence),
            FOREIGN KEY (consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE
        );
        """
        with self._lock:
            self._conn.executescript(schema)
            self._conn.execute(
                "INSERT OR IGNORE INTO coordinator_meta(key,value) VALUES"
                "('schema_version',?),('scheduler_cursor',''),"
                "('peak_retained_artifacts','0'),('peak_retained_bytes','0'),"
                "('peak_inflight_acquisitions','0')",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> WindowedArtifactCoordinator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _set_max_meta_locked(conn: sqlite3.Connection, key: str, observed: int) -> None:
        row = conn.execute(
            "SELECT value FROM coordinator_meta WHERE key=?", (key,)
        ).fetchone()
        previous = int(row["value"]) if row is not None else 0
        if observed > previous:
            conn.execute(
                "INSERT OR REPLACE INTO coordinator_meta(key,value) VALUES(?,?)",
                (key, str(observed)),
            )

    def _update_high_water_locked(self, conn: sqlite3.Connection) -> None:
        retained = conn.execute(
            "SELECT COUNT(*) AS count,COALESCE(SUM(size_bytes),0) AS bytes "
            "FROM artifacts WHERE state IN (?,?)",
            (ArtifactState.READY.value, ArtifactState.EVICTING.value),
        ).fetchone()
        inflight = int(conn.execute("SELECT COUNT(*) FROM acquisitions").fetchone()[0])
        self._set_max_meta_locked(
            conn, "peak_retained_artifacts", int(retained["count"])
        )
        self._set_max_meta_locked(conn, "peak_retained_bytes", int(retained["bytes"]))
        self._set_max_meta_locked(conn, "peak_inflight_acquisitions", inflight)

    @staticmethod
    def _validate_digest(name: str, value: str) -> None:
        if len(value) != DIGEST_LENGTH or any(
            ch not in "0123456789abcdef" for ch in value
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @staticmethod
    def _validate_consumer_id(consumer_id: str) -> None:
        if not consumer_id or len(consumer_id) > MAX_CONSUMER_ID_LENGTH:
            raise ValueError("consumer_id must contain 1-128 characters")

    def register_stream(self, contract: Mapping[str, Any]) -> str:
        contract_json = _canonical_json(contract)
        stream_id = canonical_stream_id(contract)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT contract_json FROM streams WHERE stream_id=?", (stream_id,)
            ).fetchone()
            if row is not None and row["contract_json"] != contract_json:
                raise WindowedArtifactError("stream identity collision")
            conn.execute(
                "INSERT OR IGNORE INTO streams VALUES(?,?,?)",
                (stream_id, contract_json, self._clock()),
            )
        return stream_id

    def register_positions(self, samples: Sequence[StreamSampleIndex]) -> None:
        if not samples:
            return
        stream_ids = {sample.stream_id for sample in samples}
        if len(stream_ids) != 1:
            raise ValueError("all registered positions must belong to one stream")
        stream_id = next(iter(stream_ids))
        with self._transaction() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM streams WHERE stream_id=?", (stream_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown stream {stream_id!r}")
            now = self._clock()
            for sample in samples:
                expected_position_id = canonical_position_id(
                    stream_id,
                    epoch=sample.epoch,
                    ordinal=sample.ordinal,
                    dataset_index=sample.dataset_index,
                    batch_ordinal=sample.batch_ordinal,
                    batch_start_sequence=sample.batch_start_sequence,
                    batch_end_sequence=sample.batch_end_sequence,
                )
                if sample.position_id != expected_position_id:
                    raise ValueError(
                        "sample position identity does not match its fields"
                    )
                row = conn.execute(
                    "SELECT * FROM positions WHERE stream_id=? AND sequence=?",
                    (stream_id, sample.sequence),
                ).fetchone()
                identity = (
                    sample.epoch,
                    sample.ordinal,
                    sample.dataset_index,
                    sample.batch_ordinal,
                    sample.batch_start_sequence,
                    sample.batch_end_sequence,
                    sample.request_id,
                    sample.position_id,
                )
                if row is not None:
                    observed = tuple(
                        row[name]
                        for name in (
                            "epoch",
                            "ordinal",
                            "dataset_index",
                            "batch_ordinal",
                            "batch_start_sequence",
                            "batch_end_sequence",
                            "request_id",
                            "position_id",
                        )
                    )
                    if observed != identity:
                        raise WindowedArtifactError(
                            "registered stream position changed: "
                            f"sequence={sample.sequence}"
                        )
                    continue
                conn.execute(
                    "INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        stream_id,
                        sample.sequence,
                        *identity,
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO artifacts"
                    "(request_id,state,updated_at) VALUES(?,?,?)",
                    (sample.request_id, ArtifactState.ABSENT.value, now),
                )
            consumers = conn.execute(
                "SELECT consumer_id FROM consumers WHERE stream_id=? "
                "AND state='active'",
                (stream_id,),
            ).fetchall()
            for row in consumers:
                self._refresh_window_locked(conn, row["consumer_id"])

    def register_consumer(
        self,
        consumer_id: str,
        *,
        stream_id: str,
        lookbehind: int,
        lookahead: int,
        max_prefetch: int,
        max_inflight: int,
        cursor: int = 0,
        reset: bool = False,
    ) -> None:
        self._validate_consumer_id(consumer_id)
        self._validate_digest("stream_id", stream_id)
        for name, value in (
            ("lookbehind", lookbehind),
            ("lookahead", lookahead),
            ("max_prefetch", max_prefetch),
            ("cursor", cursor),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int):
            raise TypeError("max_inflight must be an integer")
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least one")
        if max_prefetch > lookahead + 1:
            raise ValueError("max_prefetch must not exceed lookahead + 1")

        with self._transaction() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM streams WHERE stream_id=?", (stream_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown stream {stream_id!r}")
            row = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            config = (
                stream_id,
                lookbehind,
                lookahead,
                max_prefetch,
                max_inflight,
            )
            now = self._clock()
            if row is None:
                conn.execute(
                    "INSERT INTO consumers VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        consumer_id,
                        stream_id,
                        cursor,
                        lookbehind,
                        lookahead,
                        max_prefetch,
                        max_inflight,
                        "active",
                        now,
                        now,
                    ),
                )
            else:
                observed = tuple(
                    row[name]
                    for name in (
                        "stream_id",
                        "lookbehind",
                        "lookahead",
                        "max_prefetch",
                        "max_inflight",
                    )
                )
                if observed != config:
                    raise WindowedArtifactError(
                        f"consumer {consumer_id!r} configuration changed"
                    )
                if reset:
                    conn.execute(
                        "DELETE FROM acquisitions WHERE consumer_id=?",
                        (consumer_id,),
                    )
                    conn.execute(
                        "DELETE FROM completed_positions WHERE consumer_id=?",
                        (consumer_id,),
                    )
                    conn.execute(
                        "DELETE FROM interests WHERE consumer_id=?", (consumer_id,)
                    )
                    conn.execute(
                        "UPDATE consumers SET cursor=?,state='active',heartbeat_at=?,"
                        "updated_at=? WHERE consumer_id=?",
                        (cursor, now, now, consumer_id),
                    )
                elif row["state"] == "completed" and int(row["cursor"]) == cursor:
                    conn.execute(
                        "UPDATE consumers SET state='active',heartbeat_at=?,"
                        "updated_at=? "
                        "WHERE consumer_id=?",
                        (now, now, consumer_id),
                    )
                elif row["state"] != "active":
                    raise WindowedArtifactError(
                        f"consumer {consumer_id!r} is {row['state']!r}; resume "
                        "requires an explicit cursor reset"
                    )
                elif int(row["cursor"]) != cursor:
                    raise WindowedArtifactError(
                        f"consumer {consumer_id!r} cursor is {row['cursor']}, "
                        f"not requested cursor {cursor}"
                    )
                else:
                    conn.execute(
                        "UPDATE consumers SET heartbeat_at=?,updated_at=? "
                        "WHERE consumer_id=?",
                        (now, now, consumer_id),
                    )
            self._refresh_window_locked(conn, consumer_id)

    def _queue_artifact_locked(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        priority: ArtifactPriority,
    ) -> None:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO artifacts(request_id,state,updated_at) VALUES(?,?,?)",
                (request_id, ArtifactState.ABSENT.value, self._clock()),
            )
            row = conn.execute(
                "SELECT * FROM artifacts WHERE request_id=?", (request_id,)
            ).fetchone()
        state = ArtifactState(row["state"])
        if state == ArtifactState.FAILED and int(row["failures"]) >= (
            self.max_generation_attempts
        ):
            return
        if state in (ArtifactState.ABSENT, ArtifactState.FAILED):
            conn.execute(
                "UPDATE artifacts SET state=?,priority=?,queued_at=?,claim_owner=NULL,"
                "claim_until=NULL,updated_at=? WHERE request_id=?",
                (
                    ArtifactState.QUEUED.value,
                    int(priority),
                    self._clock(),
                    self._clock(),
                    request_id,
                ),
            )
        elif state == ArtifactState.QUEUED and (
            row["priority"] is None or int(row["priority"]) > int(priority)
        ):
            conn.execute(
                "UPDATE artifacts SET priority=?,updated_at=? WHERE request_id=?",
                (int(priority), self._clock(), request_id),
            )

    @staticmethod
    def _retry_priority_locked(
        conn: sqlite3.Connection, request_id: str
    ) -> ArtifactPriority:
        demand = conn.execute(
            "SELECT 1 FROM acquisitions WHERE request_id=? LIMIT 1",
            (request_id,),
        ).fetchone()
        return (
            ArtifactPriority.DEMAND if demand is not None else ArtifactPriority.PREFETCH
        )

    def _refresh_window_locked(
        self, conn: sqlite3.Connection, consumer_id: str
    ) -> None:
        consumer = conn.execute(
            "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
        ).fetchone()
        if consumer is None:
            raise KeyError(f"unknown consumer {consumer_id!r}")
        if consumer["state"] != "active":
            return
        cursor = int(consumer["cursor"])
        low = max(0, cursor - int(consumer["lookbehind"]))
        high = cursor + int(consumer["lookahead"]) + 1
        rows = conn.execute(
            "SELECT sequence,request_id FROM positions WHERE stream_id=? "
            "AND batch_end_sequence>? AND batch_start_sequence<? "
            "ORDER BY sequence",
            (consumer["stream_id"], low, high),
        ).fetchall()
        desired = {int(row["sequence"]) for row in rows}
        now = self._clock()
        for row in rows:
            conn.execute(
                "INSERT OR IGNORE INTO interests VALUES(?,?,?,?,?,?)",
                (
                    consumer_id,
                    consumer["stream_id"],
                    int(row["sequence"]),
                    row["request_id"],
                    "window",
                    now,
                ),
            )
        existing = conn.execute(
            "SELECT sequence FROM interests WHERE consumer_id=? AND kind='window'",
            (consumer_id,),
        ).fetchall()
        for row in existing:
            if int(row["sequence"]) not in desired:
                conn.execute(
                    "DELETE FROM interests WHERE consumer_id=? AND sequence=? "
                    "AND kind='window'",
                    (consumer_id, int(row["sequence"])),
                )
        self._prune_orphaned_locked(conn)
        self._top_up_prefetch_locked(conn, consumer_id)

    def _top_up_prefetch_locked(
        self, conn: sqlite3.Connection, consumer_id: str
    ) -> None:
        consumer = conn.execute(
            "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
        ).fetchone()
        if consumer is None or consumer["state"] != "active":
            return
        cap = int(consumer["max_prefetch"])
        if cap == 0:
            return
        outstanding = int(
            conn.execute(
                "SELECT COUNT(DISTINCT a.request_id) FROM interests i "
                "JOIN artifacts a ON a.request_id=i.request_id "
                "WHERE i.consumer_id=? AND i.kind='window' AND a.priority=? "
                "AND a.state IN (?,?)",
                (
                    consumer_id,
                    int(ArtifactPriority.PREFETCH),
                    ArtifactState.QUEUED.value,
                    ArtifactState.GENERATING.value,
                ),
            ).fetchone()[0]
        )
        remaining = max(0, cap - outstanding)
        if not remaining:
            return
        cursor = int(consumer["cursor"])
        high = cursor + int(consumer["lookahead"]) + 1
        candidates = conn.execute(
            "SELECT a.request_id,MIN(i.sequence) AS first_sequence "
            "FROM interests i JOIN artifacts a ON a.request_id=i.request_id "
            "WHERE i.consumer_id=? AND i.kind='window' AND i.sequence>=? "
            "AND i.sequence<? AND a.state IN (?,?) AND a.failures<? "
            "GROUP BY a.request_id ORDER BY first_sequence,a.request_id LIMIT ?",
            (
                consumer_id,
                cursor,
                high,
                ArtifactState.ABSENT.value,
                ArtifactState.FAILED.value,
                self.max_generation_attempts,
                remaining,
            ),
        ).fetchall()
        for row in candidates:
            self._queue_artifact_locked(
                conn, row["request_id"], ArtifactPriority.PREFETCH
            )

    def _prune_orphaned_locked(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE artifacts SET state=?,priority=NULL,queued_at=NULL,updated_at=? "
            "WHERE state IN (?,?) AND NOT EXISTS "
            "(SELECT 1 FROM interests i WHERE i.request_id=artifacts.request_id) "
            "AND NOT EXISTS (SELECT 1 FROM acquisitions a "
            "WHERE a.request_id=artifacts.request_id)",
            (
                ArtifactState.ABSENT.value,
                self._clock(),
                ArtifactState.QUEUED.value,
                ArtifactState.FAILED.value,
            ),
        )

    def heartbeat(self, consumer_id: str) -> None:
        with self._transaction() as conn:
            result = conn.execute(
                "UPDATE consumers SET heartbeat_at=?,updated_at=? "
                "WHERE consumer_id=? AND state='active'",
                (self._clock(), self._clock(), consumer_id),
            )
            if result.rowcount != 1:
                raise KeyError(f"unknown or inactive consumer {consumer_id!r}")

    def recover_expired(self) -> dict[str, int]:
        expired_consumers = 0
        expired_claims = 0
        with self._transaction() as conn:
            now = self._clock()
            consumers = conn.execute(
                "SELECT consumer_id FROM consumers WHERE state='active' "
                "AND heartbeat_at<?",
                (now - self.consumer_timeout_seconds,),
            ).fetchall()
            for row in consumers:
                consumer_id = row["consumer_id"]
                conn.execute(
                    "UPDATE consumers SET state='expired',updated_at=? "
                    "WHERE consumer_id=?",
                    (now, consumer_id),
                )
                conn.execute(
                    "DELETE FROM acquisitions WHERE consumer_id=?", (consumer_id,)
                )
                conn.execute(
                    "DELETE FROM interests WHERE consumer_id=?", (consumer_id,)
                )
                conn.execute(
                    "DELETE FROM completed_positions WHERE consumer_id=?",
                    (consumer_id,),
                )
                expired_consumers += 1
            claims = conn.execute(
                "SELECT request_id,failures FROM artifacts "
                "WHERE state=? AND claim_until<?",
                (ArtifactState.GENERATING.value, now),
            ).fetchall()
            for row in claims:
                request_id = row["request_id"]
                interested = conn.execute(
                    "SELECT 1 FROM interests WHERE request_id=? LIMIT 1",
                    (request_id,),
                ).fetchone()
                failures = int(row["failures"]) + 1
                retry = (
                    interested is not None and failures < self.max_generation_attempts
                )
                priority = (
                    self._retry_priority_locked(conn, request_id) if retry else None
                )
                conn.execute(
                    "UPDATE artifacts SET state=?,generation=generation+1,"
                    "failures=?,priority=?,queued_at=?,claim_owner=NULL,"
                    "claim_until=NULL,"
                    "last_error='generation claim expired',updated_at=? "
                    "WHERE request_id=?",
                    (
                        ArtifactState.QUEUED.value
                        if retry
                        else ArtifactState.FAILED.value,
                        failures,
                        int(priority) if priority is not None else None,
                        now if retry else None,
                        now,
                        request_id,
                    ),
                )
                expired_claims += 1
            self._prune_orphaned_locked(conn)
            for row in conn.execute(
                "SELECT consumer_id FROM consumers WHERE state='active'"
            ).fetchall():
                self._top_up_prefetch_locked(conn, row["consumer_id"])
        return {
            "expired_consumers": expired_consumers,
            "expired_claims": expired_claims,
        }

    def acquire(
        self,
        consumer_id: str,
        sample: StreamSampleIndex,
        *,
        timeout_seconds: float | None,
    ) -> ArtifactReadLease:
        """Wait for an authorized stream position and acquire a read lease."""
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        started = time.monotonic()
        deadline = None if timeout_seconds is None else started + timeout_seconds
        token: str | None = None
        while token is None:
            with self._transaction() as conn:
                consumer = conn.execute(
                    "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
                ).fetchone()
                if consumer is None or consumer["state"] != "active":
                    raise WindowedArtifactError(
                        f"consumer {consumer_id!r} is not active"
                    )
                position = conn.execute(
                    "SELECT * FROM positions WHERE stream_id=? AND sequence=?",
                    (sample.stream_id, sample.sequence),
                ).fetchone()
                expected = (
                    sample.epoch,
                    sample.ordinal,
                    sample.dataset_index,
                    sample.batch_ordinal,
                    sample.batch_start_sequence,
                    sample.batch_end_sequence,
                    sample.request_id,
                    sample.position_id,
                )
                observed = (
                    tuple(
                        position[name]
                        for name in (
                            "epoch",
                            "ordinal",
                            "dataset_index",
                            "batch_ordinal",
                            "batch_start_sequence",
                            "batch_end_sequence",
                            "request_id",
                            "position_id",
                        )
                    )
                    if position is not None
                    else None
                )
                if consumer["stream_id"] != sample.stream_id or observed != expected:
                    raise WindowedArtifactError(
                        "requested sample does not match the registered stream position"
                    )
                now = self._clock()
                conn.execute(
                    "UPDATE consumers SET heartbeat_at=?,updated_at=? "
                    "WHERE consumer_id=?",
                    (now, now, consumer_id),
                )
                cursor = int(consumer["cursor"])
                low = max(0, cursor - int(consumer["lookbehind"]))
                high = cursor + int(consumer["lookahead"])
                inflight = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM acquisitions WHERE consumer_id=?",
                        (consumer_id,),
                    ).fetchone()[0]
                )
                same_batch = (
                    conn.execute(
                        "SELECT 1 FROM acquisitions a JOIN positions p "
                        "ON p.stream_id=a.stream_id AND p.sequence=a.sequence "
                        "WHERE a.consumer_id=? AND p.epoch=? "
                        "AND p.batch_ordinal=? LIMIT 1",
                        (consumer_id, sample.epoch, sample.batch_ordinal),
                    ).fetchone()
                    is not None
                )
                inside_window = (
                    sample.batch_end_sequence > low
                    and sample.batch_start_sequence <= high
                )
                has_capacity = inflight < int(consumer["max_inflight"])
                if inside_window and (has_capacity or same_batch):
                    token = uuid.uuid4().hex
                    conn.execute(
                        "INSERT INTO acquisitions VALUES(?,?,?,?,?,'waiting',?,?)",
                        (
                            token,
                            consumer_id,
                            sample.stream_id,
                            sample.sequence,
                            sample.request_id,
                            now,
                            now,
                        ),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO interests VALUES(?,?,?,?,?,?)",
                        (
                            consumer_id,
                            sample.stream_id,
                            sample.sequence,
                            sample.request_id,
                            "demand",
                            now,
                        ),
                    )
                    self._queue_artifact_locked(
                        conn, sample.request_id, ArtifactPriority.DEMAND
                    )
                    self._update_high_water_locked(conn)
            if token is None:
                self._sleep_or_timeout(deadline, started, timeout_seconds, sample)

        while True:
            with self._transaction() as conn:
                acquisition = conn.execute(
                    "SELECT * FROM acquisitions WHERE token=?", (token,)
                ).fetchone()
                if acquisition is None:
                    raise WindowedArtifactError(
                        f"artifact acquisition {token} was released while waiting"
                    )
                artifact = conn.execute(
                    "SELECT * FROM artifacts WHERE request_id=?",
                    (sample.request_id,),
                ).fetchone()
                if artifact is None:
                    raise WindowedArtifactError("artifact metadata disappeared")
                state = ArtifactState(artifact["state"])
                if state == ArtifactState.READY:
                    if not artifact["path"]:
                        raise WindowedArtifactError("ready artifact has no path")
                    cache_hit = bool(artifact["first_reader_accounted"])
                    conn.execute(
                        "UPDATE artifacts SET first_reader_accounted=1,updated_at=? "
                        "WHERE request_id=?",
                        (self._clock(), sample.request_id),
                    )
                    conn.execute(
                        "UPDATE acquisitions SET state='leased',updated_at=? "
                        "WHERE token=?",
                        (self._clock(), token),
                    )
                    return ArtifactReadLease(
                        token=token,
                        consumer_id=consumer_id,
                        stream_id=sample.stream_id,
                        sequence=sample.sequence,
                        request_id=sample.request_id,
                        path=Path(artifact["path"]),
                        generation=int(artifact["generation"]),
                        cache_hit=cache_hit,
                        wait_seconds=time.monotonic() - started,
                    )
                if state == ArtifactState.FAILED and int(artifact["failures"]) >= (
                    self.max_generation_attempts
                ):
                    conn.execute("DELETE FROM acquisitions WHERE token=?", (token,))
                    conn.execute(
                        "DELETE FROM interests WHERE consumer_id=? AND stream_id=? "
                        "AND sequence=? AND kind='demand'",
                        (consumer_id, sample.stream_id, sample.sequence),
                    )
                    raise ArtifactGenerationError(
                        f"artifact {sample.request_id} failed after "
                        f"{artifact['failures']} attempts: {artifact['last_error']}"
                    )
                if state == ArtifactState.FAILED:
                    self._queue_artifact_locked(
                        conn, sample.request_id, ArtifactPriority.DEMAND
                    )
                conn.execute(
                    "UPDATE consumers SET heartbeat_at=?,updated_at=? "
                    "WHERE consumer_id=?",
                    (self._clock(), self._clock(), consumer_id),
                )
            self._sleep_or_timeout(deadline, started, timeout_seconds, sample, token)

    def _sleep_or_timeout(
        self,
        deadline: float | None,
        started: float,
        timeout_seconds: float | None,
        sample: StreamSampleIndex,
        token: str | None = None,
    ) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            if token is not None:
                self.abandon_tokens(sample.stream_id, [token])
            raise TimeoutError(
                f"stream position {sample.sequence} was not ready within "
                f"{timeout_seconds:.1f}s"
            )
        sleep_seconds = self.poll_seconds
        if deadline is not None:
            sleep_seconds = min(sleep_seconds, max(0.0, deadline - time.monotonic()))
        if time.monotonic() >= started:
            time.sleep(sleep_seconds)

    def ack(self, consumer_id: str, leases: Sequence[Mapping[str, Any]]) -> int:
        """Commit successful trainer consumption and advance a contiguous cursor."""
        if not leases:
            with self._lock:
                row = self._conn.execute(
                    "SELECT cursor FROM consumers WHERE consumer_id=?", (consumer_id,)
                ).fetchone()
            if row is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            return int(row["cursor"])
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None or consumer["state"] != "active":
                raise WindowedArtifactError(f"consumer {consumer_id!r} is not active")
            for lease in leases:
                token = str(lease["token"])
                row = conn.execute(
                    "SELECT * FROM acquisitions WHERE token=?", (token,)
                ).fetchone()
                expected = (
                    consumer_id,
                    str(lease["stream_id"]),
                    int(lease["sequence"]),
                    str(lease["request_id"]),
                    "leased",
                )
                observed = (
                    (
                        row["consumer_id"],
                        row["stream_id"],
                        int(row["sequence"]),
                        row["request_id"],
                        row["state"],
                    )
                    if row is not None
                    else None
                )
                if observed != expected:
                    raise WindowedArtifactError(
                        f"read lease {token!r} is unknown, stale, or mismatched"
                    )
                conn.execute(
                    "INSERT OR IGNORE INTO completed_positions VALUES(?,?)",
                    (consumer_id, int(row["sequence"])),
                )
                conn.execute("DELETE FROM acquisitions WHERE token=?", (token,))
                conn.execute(
                    "DELETE FROM interests WHERE consumer_id=? AND stream_id=? "
                    "AND sequence=? AND kind='demand'",
                    (consumer_id, row["stream_id"], int(row["sequence"])),
                )
            cursor = int(consumer["cursor"])
            while (
                conn.execute(
                    "SELECT 1 FROM completed_positions WHERE consumer_id=? "
                    "AND sequence=?",
                    (consumer_id, cursor),
                ).fetchone()
                is not None
            ):
                conn.execute(
                    "DELETE FROM completed_positions WHERE consumer_id=? "
                    "AND sequence=?",
                    (consumer_id, cursor),
                )
                cursor += 1
            now = self._clock()
            conn.execute(
                "UPDATE consumers SET cursor=?,heartbeat_at=?,updated_at=? "
                "WHERE consumer_id=?",
                (cursor, now, now, consumer_id),
            )
            self._refresh_window_locked(conn, consumer_id)
            return cursor

    def abandon(self, consumer_id: str, leases: Sequence[Mapping[str, Any]]) -> None:
        self._abandon_tokens(consumer_id, [str(lease["token"]) for lease in leases])

    def abandon_tokens(self, stream_id: str, tokens: Sequence[str]) -> None:
        """Release timed-out waiters when only the stream identity is available."""
        del stream_id
        self._abandon_tokens(None, tokens)

    def _abandon_tokens(self, consumer_id: str | None, tokens: Sequence[str]) -> None:
        if not tokens:
            return
        with self._transaction() as conn:
            affected: set[str] = set()
            for token in tokens:
                row = conn.execute(
                    "SELECT * FROM acquisitions WHERE token=?", (token,)
                ).fetchone()
                if row is None:
                    continue
                if consumer_id is not None and row["consumer_id"] != consumer_id:
                    raise WindowedArtifactError(
                        f"lease {token!r} does not belong to {consumer_id!r}"
                    )
                affected.add(row["consumer_id"])
                conn.execute("DELETE FROM acquisitions WHERE token=?", (token,))
                conn.execute(
                    "DELETE FROM interests WHERE consumer_id=? AND stream_id=? "
                    "AND sequence=? AND kind='demand'",
                    (row["consumer_id"], row["stream_id"], int(row["sequence"])),
                )
            for owner in affected:
                self._refresh_window_locked(conn, owner)

    def claim_generation(
        self,
        owner: str,
        *,
        stream_id: str,
        max_claims: int = 1,
        max_active_claims: int | None = None,
    ) -> tuple[GenerationClaim, ...]:
        if not owner:
            raise ValueError("generation owner must be non-empty")
        if max_claims < 1:
            raise ValueError("max_claims must be at least one")
        if max_active_claims is not None and max_active_claims < 1:
            raise ValueError("max_active_claims must be at least one")
        with self._transaction() as conn:
            self._recover_claims_locked(conn)
            if max_active_claims is not None:
                active = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM artifacts WHERE state=?",
                        (ArtifactState.GENERATING.value,),
                    ).fetchone()[0]
                )
                max_claims = min(max_claims, max_active_claims - active)
                if max_claims <= 0:
                    return ()
            rows = conn.execute(
                "SELECT DISTINCT a.* FROM artifacts a "
                "JOIN interests i ON i.request_id=a.request_id "
                "WHERE a.state=? AND i.stream_id=? "
                "ORDER BY a.priority,a.queued_at,a.request_id LIMIT ?",
                (
                    ArtifactState.QUEUED.value,
                    stream_id,
                    max(64, max_claims * 8),
                ),
            ).fetchall()
            if not rows:
                return ()
            prefetch_rows: list[sqlite3.Row] = []
            by_consumer: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                consumers = conn.execute(
                    "SELECT DISTINCT consumer_id FROM acquisitions "
                    "WHERE request_id=? ORDER BY consumer_id",
                    (row["request_id"],),
                ).fetchall()
                is_demand = int(row["priority"]) == int(
                    ArtifactPriority.DEMAND
                ) and bool(consumers)
                if is_demand:
                    for consumer in consumers:
                        by_consumer.setdefault(consumer["consumer_id"], []).append(row)
                else:
                    prefetch_rows.append(row)
            cursor_row = conn.execute(
                "SELECT value FROM coordinator_meta WHERE key='scheduler_cursor'"
            ).fetchone()
            previous = cursor_row["value"] if cursor_row is not None else ""
            consumer_order = sorted(by_consumer)
            if previous in consumer_order:
                start = (consumer_order.index(previous) + 1) % len(consumer_order)
                consumer_order = consumer_order[start:] + consumer_order[:start]

            selected: list[sqlite3.Row] = []
            selected_ids: set[str] = set()
            last_consumer = previous
            while consumer_order and len(selected) < max_claims:
                made_progress = False
                for consumer_id in consumer_order:
                    while by_consumer[consumer_id]:
                        row = by_consumer[consumer_id].pop(0)
                        if row["request_id"] not in selected_ids:
                            selected.append(row)
                            selected_ids.add(row["request_id"])
                            last_consumer = consumer_id
                            made_progress = True
                            break
                    if len(selected) >= max_claims:
                        break
                if not made_progress:
                    break
            for row in prefetch_rows:
                if len(selected) >= max_claims:
                    break
                if row["request_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["request_id"])

            now = self._clock()
            claims: list[GenerationClaim] = []
            for row in selected:
                position = conn.execute(
                    "SELECT dataset_index FROM positions WHERE stream_id=? "
                    "AND request_id=? ORDER BY sequence LIMIT 1",
                    (stream_id, row["request_id"]),
                ).fetchone()
                if position is None:
                    continue
                result = conn.execute(
                    "UPDATE artifacts SET state=?,claim_owner=?,claim_until=?,"
                    "updated_at=? WHERE request_id=? AND state=?",
                    (
                        ArtifactState.GENERATING.value,
                        owner,
                        now + self.claim_timeout_seconds,
                        now,
                        row["request_id"],
                        ArtifactState.QUEUED.value,
                    ),
                )
                if result.rowcount != 1:
                    continue
                claims.append(
                    GenerationClaim(
                        request_id=row["request_id"],
                        stream_id=stream_id,
                        dataset_index=int(position["dataset_index"]),
                        generation=int(row["generation"]),
                        priority=ArtifactPriority(int(row["priority"])),
                    )
                )
            if last_consumer:
                conn.execute(
                    "UPDATE coordinator_meta SET value=? WHERE key='scheduler_cursor'",
                    (last_consumer,),
                )
            return tuple(claims)

    def _recover_claims_locked(self, conn: sqlite3.Connection) -> None:
        now = self._clock()
        rows = conn.execute(
            "SELECT request_id,failures FROM artifacts WHERE state=? AND claim_until<?",
            (ArtifactState.GENERATING.value, now),
        ).fetchall()
        for row in rows:
            affected = conn.execute(
                "SELECT DISTINCT consumer_id FROM interests WHERE request_id=?",
                (row["request_id"],),
            ).fetchall()
            interested = conn.execute(
                "SELECT 1 FROM interests WHERE request_id=? LIMIT 1",
                (row["request_id"],),
            ).fetchone()
            failures = int(row["failures"]) + 1
            retry = interested is not None and failures < self.max_generation_attempts
            priority = (
                self._retry_priority_locked(conn, row["request_id"]) if retry else None
            )
            conn.execute(
                "UPDATE artifacts SET state=?,generation=generation+1,failures=?,"
                "priority=?,queued_at=?,claim_owner=NULL,claim_until=NULL,"
                "last_error='generation claim expired',updated_at=? "
                "WHERE request_id=?",
                (
                    ArtifactState.QUEUED.value if retry else ArtifactState.FAILED.value,
                    failures,
                    int(priority) if priority is not None else None,
                    now if retry else None,
                    now,
                    row["request_id"],
                ),
            )
            for consumer in affected:
                self._top_up_prefetch_locked(conn, consumer["consumer_id"])

    def complete_generation(
        self,
        owner: str,
        claim: GenerationClaim,
        *,
        path: str | os.PathLike[str],
        size_bytes: int,
    ) -> None:
        artifact_path = str(Path(path).expanduser().resolve())
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE request_id=?", (claim.request_id,)
            ).fetchone()
            if row is None or (
                row["state"],
                row["claim_owner"],
                int(row["generation"]),
            ) != (ArtifactState.GENERATING.value, owner, claim.generation):
                raise WindowedArtifactError(
                    f"stale generation completion for {claim.request_id}"
                )
            conn.execute(
                "UPDATE artifacts SET state=?,path=?,size_bytes=?,priority=NULL,"
                "queued_at=NULL,claim_owner=NULL,claim_until=NULL,failures=0,"
                "last_error=NULL,first_reader_accounted=0,updated_at=? "
                "WHERE request_id=?",
                (
                    ArtifactState.READY.value,
                    artifact_path,
                    size_bytes,
                    self._clock(),
                    claim.request_id,
                ),
            )
            self._update_high_water_locked(conn)
            interested = conn.execute(
                "SELECT DISTINCT consumer_id FROM interests WHERE request_id=?",
                (claim.request_id,),
            ).fetchall()
            for consumer in interested:
                self._top_up_prefetch_locked(conn, consumer["consumer_id"])

    def fail_generation(
        self, owner: str, claim: GenerationClaim, error: BaseException | str
    ) -> None:
        message = str(error)[:2000] or type(error).__name__
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE request_id=?", (claim.request_id,)
            ).fetchone()
            if row is None or (
                row["state"],
                row["claim_owner"],
                int(row["generation"]),
            ) != (ArtifactState.GENERATING.value, owner, claim.generation):
                raise WindowedArtifactError(
                    f"stale generation failure for {claim.request_id}"
                )
            failures = int(row["failures"]) + 1
            affected = conn.execute(
                "SELECT DISTINCT consumer_id FROM interests WHERE request_id=?",
                (claim.request_id,),
            ).fetchall()
            interested = conn.execute(
                "SELECT 1 FROM interests WHERE request_id=? LIMIT 1",
                (claim.request_id,),
            ).fetchone()
            retry = failures < self.max_generation_attempts and interested is not None
            priority = (
                self._retry_priority_locked(conn, claim.request_id) if retry else None
            )
            conn.execute(
                "UPDATE artifacts SET state=?,generation=generation+1,failures=?,"
                "last_error=?,priority=?,queued_at=?,claim_owner=NULL,"
                "claim_until=NULL,updated_at=? WHERE request_id=?",
                (
                    ArtifactState.QUEUED.value if retry else ArtifactState.FAILED.value,
                    failures,
                    message,
                    int(priority) if priority is not None else None,
                    self._clock() if retry else None,
                    self._clock(),
                    claim.request_id,
                ),
            )
            for consumer in affected:
                self._top_up_prefetch_locked(conn, consumer["consumer_id"])

    def begin_evictions(self, *, limit: int = 64) -> tuple[EvictionClaim, ...]:
        if limit < 1:
            raise ValueError("eviction limit must be at least one")
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts a WHERE a.state=? "
                "AND NOT EXISTS (SELECT 1 FROM interests i "
                "WHERE i.request_id=a.request_id) "
                "AND NOT EXISTS (SELECT 1 FROM acquisitions q "
                "WHERE q.request_id=a.request_id) "
                "ORDER BY a.updated_at LIMIT ?",
                (ArtifactState.READY.value, limit),
            ).fetchall()
            claims: list[EvictionClaim] = []
            for row in rows:
                if not row["path"]:
                    continue
                result = conn.execute(
                    "UPDATE artifacts SET state=?,updated_at=? "
                    "WHERE request_id=? AND state=?",
                    (
                        ArtifactState.EVICTING.value,
                        self._clock(),
                        row["request_id"],
                        ArtifactState.READY.value,
                    ),
                )
                if result.rowcount == 1:
                    claims.append(
                        EvictionClaim(
                            request_id=row["request_id"],
                            generation=int(row["generation"]),
                            path=Path(row["path"]),
                        )
                    )
            return tuple(claims)

    def finish_eviction(self, claim: EvictionClaim, *, removed: bool) -> None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE request_id=?", (claim.request_id,)
            ).fetchone()
            if row is None or (row["state"], int(row["generation"])) != (
                ArtifactState.EVICTING.value,
                claim.generation,
            ):
                raise WindowedArtifactError(
                    f"stale eviction completion for {claim.request_id}"
                )
            interested = conn.execute(
                "SELECT DISTINCT consumer_id,kind FROM interests WHERE request_id=?",
                (claim.request_id,),
            ).fetchall()
            if removed:
                demand = any(item["kind"] == "demand" for item in interested)
                state = (
                    ArtifactState.QUEUED.value if demand else ArtifactState.ABSENT.value
                )
                priority = ArtifactPriority.DEMAND if demand else None
                conn.execute(
                    "UPDATE artifacts SET state=?,path=NULL,size_bytes=0,priority=?,"
                    "queued_at=?,updated_at=? WHERE request_id=?",
                    (
                        state,
                        int(priority) if priority is not None else None,
                        self._clock() if demand else None,
                        self._clock(),
                        claim.request_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE artifacts SET state=?,updated_at=? WHERE request_id=?",
                    (ArtifactState.READY.value, self._clock(), claim.request_id),
                )
            for consumer_id in {item["consumer_id"] for item in interested}:
                self._top_up_prefetch_locked(conn, consumer_id)

    def complete_consumer(self, consumer_id: str) -> None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            conn.execute("DELETE FROM acquisitions WHERE consumer_id=?", (consumer_id,))
            conn.execute("DELETE FROM interests WHERE consumer_id=?", (consumer_id,))
            conn.execute(
                "DELETE FROM completed_positions WHERE consumer_id=?", (consumer_id,)
            )
            conn.execute(
                "UPDATE consumers SET state='completed',updated_at=? "
                "WHERE consumer_id=?",
                (self._clock(), consumer_id),
            )
            self._prune_orphaned_locked(conn)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            consumers = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT * FROM consumers ORDER BY consumer_id"
                ).fetchall()
            ]
            artifact_states = {
                row["state"]: int(row["count"])
                for row in self._conn.execute(
                    "SELECT state,COUNT(*) AS count FROM artifacts GROUP BY state"
                ).fetchall()
            }
            totals = self._conn.execute(
                "SELECT COUNT(*) AS count,COALESCE(SUM(size_bytes),0) AS bytes "
                "FROM artifacts WHERE state IN (?,?)",
                (ArtifactState.READY.value, ArtifactState.EVICTING.value),
            ).fetchone()
            inflight = int(
                self._conn.execute("SELECT COUNT(*) FROM acquisitions").fetchone()[0]
            )
            positions = int(
                self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
            )
            high_water = {
                row["key"].removeprefix("peak_"): int(row["value"])
                for row in self._conn.execute(
                    "SELECT key,value FROM coordinator_meta WHERE key LIKE 'peak_%'"
                ).fetchall()
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "positions": positions,
            "consumers": consumers,
            "artifact_states": artifact_states,
            "retained_artifacts": int(totals["count"]),
            "retained_bytes": int(totals["bytes"]),
            "inflight_acquisitions": inflight,
            "high_water": high_water,
        }
