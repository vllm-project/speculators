#!/usr/bin/env python3
"""Tiny static file server for HTTP-based hidden-states transfer.

Run this on each vLLM node, pointing ``--root`` at the same local directory
the connector writes to (``shared_storage_path`` / ``--hidden-states-path``).
It pairs with the ``http`` hidden-states backend
(``hs_connectors.transfer.HttpBackend``): the trainer does

    GET  http://<host>:<port>/<basename>.safetensors   -> wait for write, stream
    DELETE http://<host>:<port>/<basename>.safetensors  -> drop file + lock

Why it exists
------------
The connector (``example_hidden_states_connector.py``) returns a request's
handle *before* its async DtoH copy + disk write finishes. To hide that race it
holds an advisory flock (``LOCK_EX``) on a companion ``<file>.lock`` for the
duration of the write and releases it (closes the fd) once the write is done.
The file backend's reader re-acquires that lock to block until the write
completes. Over HTTP the trainer can no longer see the lock, so this server
re-acquires it on the trainer's behalf: a ``GET`` blocks until the lock is
released (= write complete), then streams the file. This preserves the exact
synchronization semantics of the file backend with zero connector changes.

A background TTL sweeper deletes stale ``.safetensors`` files (and their
``.lock`` companions) older than ``--ttl`` seconds, so a trainer crash can never
fill the local disk with orphaned files.

Usage
-----
    python scripts/serve_hs.py --root /data/local_hs --port 9010 --host 0.0.0.0

Examples
--------
    # One server per vLLM instance, each serving that instance's own subdir:
    python scripts/serve_hs.py --root /data/local_hs/8010 --port 9010
    python scripts/serve_hs.py --root /data/local_hs/8020 --port 9020
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.server
import os
import socketserver
import threading
import time
from pathlib import Path

DEFAULT_ROOT = "/tmp/hidden_states"  # noqa: S108
DEFAULT_LOCK_TIMEOUT = 120.0  # seconds to wait for the connector's write
DEFAULT_TTL = 600.0  # seconds; stale files older than this are swept
DEFAULT_TTL_INTERVAL = 60.0  # seconds between sweeper passes
CHUNK = 1024 * 1024  # 1 MiB streaming buffer


def _wait_for_file(path: Path, timeout: float, poll: float = 0.05) -> bool:
    """Spin until ``path`` exists or ``timeout`` elapses. True if it appeared."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(poll)
    return path.exists()


def _wait_for_write(data_path: Path, lock_timeout: float) -> str | None:
    """Block until the connector finishes writing ``data_path``.

    Returns a short status string for logging ("locked"), or ``None`` if the
    file is not available within the timeout.

    Readiness is driven entirely by the connector's ``.lock`` flock: the
    connector creates ``<data_path>.lock`` and holds LOCK_EX while writing,
    then closes its fd (releasing the lock) once done. We wait for the lock
    file to appear, then flock-wait (LOCK_EX) until we acquire it — acquiring
    it means the writer released it = write done. This is race-free and never
    serves a half-written file.

    There is deliberately no fixed cap on waiting for the lock file to appear:
    it returns as soon as the lock shows up (normally within a scheduler step).
    An earlier version capped this wait at 5s then fell back to serving the
    data file by existence — which both burned a flat ~5s on every fetch
    (observed: ``get_generated`` p50 == 5035ms) AND risked serving a
    half-written file. We now just wait for the lock.
    """
    lock_path = Path(str(data_path) + ".lock")

    # Wait for the lock file to appear (connector creates it when the write
    # starts). No fixed cap — returns as soon as it appears.
    if not _wait_for_file(lock_path, timeout=lock_timeout):
        return None  # lock never appeared; do NOT serve (would risk a partial read)

    fd = os.open(str(lock_path), os.O_RDONLY)
    deadline = time.monotonic() + lock_timeout
    try:
        while time.monotonic() < deadline:
            try:
                # NB: LOCK_EX (not SH) so it blocks until the writer's exclusive
                # lock is released, exactly like the file backend's reader.
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return "locked"
            except BlockingIOError:
                time.sleep(0.02)
        return None  # timed out waiting for the writer to release
    finally:
        os.close(fd)  # releases our (just-acquired) lock; leave the file in place


def _sweep(root: Path, ttl: float, interval: float) -> None:
    """Daemon loop: delete ``.safetensors`` (and ``.lock``) older than ``ttl``."""
    while True:
        time.sleep(interval)
        now = time.monotonic()
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if now - mtime < ttl:
                continue
            if entry.suffix in {".safetensors", ".lock"}:
                with contextlib.suppress(OSError):
                    entry.unlink()


class HiddenStatesHandler(http.server.BaseHTTPRequestHandler):
    root: Path  # injected via the server factory below
    lock_timeout: float

    # Quieter logging: one line per request, no noise to stderr per default.
    def log_message(self, fmt: str, *args) -> None:
        pass

    # --- helpers ----------------------------------------------------------
    def _resolve(self, urlpath: str) -> Path | None:
        """Map URL path to a safe absolute path under ``root`` (no traversal)."""
        # Strip query/fragment, take the basename only. The connector writes a
        # flat ``{req_id}.safetensors``, so we never serve nested paths.
        name = os.path.basename(urlpath.partition("?")[0])
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            return None
        candidate = (self.root / name).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate

    # --- GET --------------------------------------------------------------
    def do_GET(self) -> None:
        data_path = self._resolve(self.path)
        if data_path is None or data_path.suffix != ".safetensors":
            self.send_error(404, "Not found")
            return

        status = _wait_for_write(data_path, self.lock_timeout)
        if status is None or not data_path.exists():
            self.send_error(404, "Hidden states not available")
            return

        try:
            size = data_path.stat().st_size
        except OSError:
            self.send_error(404, "Not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        try:
            with open(data_path, "rb") as f:
                while True:
                    buf = f.read(CHUNK)
                    if not buf:
                        break
                    self.wfile.write(buf)
        except FileNotFoundError:
            # Lost a race with DELETE / TTL sweeper after the lock-wait.
            self.send_error(404, "Not found")
        except (BrokenPipeError, ConnectionResetError):
            # Client went away mid-transfer; nothing to do.
            pass

    # --- DELETE -----------------------------------------------------------
    def do_DELETE(self) -> None:
        data_path = self._resolve(self.path)
        if data_path is None or data_path.suffix != ".safetensors":
            self.send_error(404, "Not found")
            return
        lock_path = Path(str(data_path) + ".lock")
        for p in (data_path, lock_path):
            with contextlib.suppress(OSError):
                p.unlink()
        self.send_response(204)
        self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """One thread per request so concurrent GETs don't serialize."""

    daemon_threads = True
    allow_reuse_address = True


def make_handler(root: Path, lock_timeout: float) -> type[HiddenStatesHandler]:
    return type(
        "BoundHiddenStatesHandler",
        (HiddenStatesHandler,),
        {"root": root, "lock_timeout": lock_timeout},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve connector-written hidden-states files over HTTP for the "
            "'http' hs_connectors backend. Blocks on the connector's .lock "
            "flock until each write is complete, then streams the file."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help=(
            "Directory the connector writes hidden states to (its "
            "shared_storage_path / --hidden-states-path). Must be local fast "
            f"disk. Default: {DEFAULT_ROOT}"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # noqa: S104 - must be reachable from the trainer node
        help="Bind address. Default: 0.0.0.0 (all interfaces).",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT,
        help=(
            "Seconds to wait for the connector to finish writing a file before "
            f"giving up (GET). Default: {DEFAULT_LOCK_TIMEOUT}"
        ),
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=DEFAULT_TTL,
        help=(
            "Stale .safetensors/.lock files older than this many seconds are "
            f"deleted by the background sweeper. Default: {DEFAULT_TTL}"
        ),
    )
    parser.add_argument(
        "--ttl-interval",
        type=float,
        default=DEFAULT_TTL_INTERVAL,
        help=f"Seconds between sweeper passes. Default: {DEFAULT_TTL_INTERVAL}",
    )
    parser.add_argument(
        "--no-sweeper",
        action="store_true",
        help="Disable the background TTL sweeper (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"[serve_hs] root={root} host={args.host} port={args.port} "
        f"lock_timeout={args.lock_timeout}s ttl={args.ttl}s",
        flush=True,
    )

    if not args.no_sweeper:
        t = threading.Thread(
            target=_sweep,
            args=(root, args.ttl, args.ttl_interval),
            daemon=True,
        )
        t.start()

    handler = make_handler(root, args.lock_timeout)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
