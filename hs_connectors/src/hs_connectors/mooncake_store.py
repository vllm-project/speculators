"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.

Each sample is packed into one contiguous self-describing buffer
(``[8-byte header length][JSON tensor layout][tensor bytes]``) and moved with a
single zero-copy operation: the producer stages tensors into a pinned,
Mooncake-registered host buffer and calls ``put_from``; the consumer calls
``get_buffer`` and reconstructs the tensors from the returned buffer. One store
op per sample instead of one per tensor — round-trips, not bandwidth, dominate
transfer cost over TCP.
"""

from __future__ import annotations

import ctypes
import inspect
import json
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass

import torch


def resolve_local_hostname(master_server_address: str) -> str:
    """Routable address for this client, as seen on the route to the master.

    ``gethostbyname(gethostname())`` resolves to loopback on many hosts, which
    breaks cross-node transfers. A UDP socket toward the master (no packets
    sent) yields the interface the OS actually routes through. Override with
    ``MOONCAKE_LOCAL_HOSTNAME`` if auto-detection picks the wrong interface.
    """
    override = os.environ.get("MOONCAKE_LOCAL_HOSTNAME")
    if override:
        return override
    host = master_server_address.rsplit(":", 1)[0]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, 1))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _ensure_remove_force(store) -> None:
    """Fail fast if the mooncake build lacks ``remove(key, force=True)``.

    Without ``force=`` deletes fail, and the store silently fills under load
    until producer puts stall. Requires mooncake >= 0.3.10.post1; checked at
    runtime because the ``mooncake-transfer-engine-cuda13`` wheel's package
    name evades the setup.py pin.
    """
    try:
        has_force = "force" in inspect.signature(store.remove).parameters
    except (TypeError, ValueError):  # pybind methods aren't introspectable
        has_force = "force" in (getattr(store.remove, "__doc__", None) or "")
    if not has_force:
        raise RuntimeError(
            "mooncake-transfer-engine >= 0.3.10.post1 is required: this build's "
            "remove() lacks force=, so consumed samples can never be deleted "
            "and the store would fill until puts stall."
        )


# Process-wide DtoH staging stream, mirroring vLLM's aux_stream pattern:
# lazily created once and reused for the process lifetime.
_staging_stream: torch.cuda.Stream | None = None
_staging_stream_lock = threading.Lock()


def _get_staging_stream() -> torch.cuda.Stream:
    global _staging_stream  # noqa: PLW0603
    if _staging_stream is None:
        with _staging_stream_lock:
            if _staging_stream is None:
                _staging_stream = torch.cuda.Stream()
    return _staging_stream


def _pack_into(
    buf: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    copy_stream: torch.cuda.Stream | None = None,
) -> int:
    """Pack ``tensors`` into ``buf`` (pinned uint8) and return the byte size.

    When ``copy_stream`` is given the caller must already be inside that stream
    context and synchronize it afterwards: GPU->host copies are issued
    asynchronously, with ``record_stream`` guarding the source tensors.
    """
    layout: dict[str, dict] = {}
    staged: list[tuple[int, torch.Tensor]] = []
    offset = 0
    for name, tensor in tensors.items():
        t = tensor.detach().contiguous()
        nbytes = t.numel() * t.element_size()
        layout[name] = {
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "offset": offset,
            "nbytes": nbytes,
        }
        staged.append((offset, t))
        offset += nbytes

    header = json.dumps(layout).encode("utf-8")
    prefix = len(header).to_bytes(8, "little") + header
    total = len(prefix) + offset
    if total > buf.numel():
        raise RuntimeError(
            f"sample ({total} bytes) exceeds transfer_buffer_size ({buf.numel()} "
            "bytes); increase MooncakeStoreConfig.transfer_buffer_size"
        )

    buf[: len(prefix)].copy_(torch.frombuffer(bytearray(prefix), dtype=torch.uint8))
    for off, t in staged:
        start = len(prefix) + off
        nbytes = t.numel() * t.element_size()
        async_copy = copy_stream is not None and t.is_cuda
        buf[start : start + nbytes].copy_(
            t.view(torch.uint8).reshape(-1), non_blocking=async_copy
        )
        if async_copy:
            t.record_stream(copy_stream)
    return total


def _unpack_from(ptr: int, size: int) -> dict[str, torch.Tensor]:
    """Reconstruct a tensor dict from a packed buffer at ``ptr``.

    Tensors are copied out into freshly allocated memory so the result stays
    valid after the Mooncake-managed source buffer is reused.
    """
    header_len = int.from_bytes(ctypes.string_at(ptr, 8), "little")
    layout = json.loads(ctypes.string_at(ptr + 8, header_len))
    data_start = 8 + header_len

    raw = torch.frombuffer((ctypes.c_char * size).from_address(ptr), dtype=torch.uint8)
    out: dict[str, torch.Tensor] = {}
    for name, entry in layout.items():
        t = torch.empty(entry["shape"], dtype=getattr(torch, entry["dtype"]))
        start = data_start + entry["offset"]
        t.view(torch.uint8).reshape(-1).copy_(raw[start : start + entry["nbytes"]])
        out[name] = t
    return out


class _PinnedBufferPool:
    """Fixed pool of pinned, Mooncake-registered host buffers for ``put_from``."""

    def __init__(self, store, buffer_size: int, pool_size: int):
        self._free: queue.Queue[torch.Tensor] = queue.Queue()
        pin = torch.cuda.is_available()  # pinning requires a CUDA context
        for _ in range(pool_size):
            buf = torch.empty(buffer_size, dtype=torch.uint8, pin_memory=pin)
            rc = store.register_buffer(buf.data_ptr(), buffer_size)
            if rc != 0:
                raise RuntimeError(f"mooncake register_buffer failed (rc={rc})")
            self._free.put(buf)

    def acquire(self) -> torch.Tensor:
        return self._free.get()

    def release(self, buf: torch.Tensor) -> None:
        self._free.put(buf)


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8080/metadata"
    master_server_address: str = "localhost:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 2 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 16
    # Producer-side transfer buffers: transfer_buffer_size must hold the
    # largest packed sample; the pool lets staging overlap the (serialized)
    # put_from calls, so a small pool suffices.
    transfer_buffer_size: int = 256 * 1024 * 1024
    transfer_pool_size: int = 4

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def for_consumer(cls, **kwargs) -> MooncakeStoreConfig:
        """Config for read-only clients (trainer dataloader workers).

        Consumers mount no segment (``global_segment_size=0``): samples only
        land in producer segments, and worker processes avoid registering GBs
        of memory each. The local buffer only covers in-flight reads.
        """
        kwargs.setdefault("global_segment_size", 0)
        kwargs.setdefault("local_buffer_size", 512 * 1024 * 1024)
        return cls(**kwargs)


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store via zero-copy buffers."""

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None
        self._put_pool: _PinnedBufferPool | None = None
        self._put_lock = threading.Lock()

    @property
    def is_setup(self):
        return self._store is not None

    def setup(self) -> MooncakeHiddenStatesStore:
        if self._store is not None:
            return self
        try:
            from mooncake.store import (  # type: ignore[import-not-found] # noqa: PLC0415
                MooncakeDistributedStore,
            )
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Mooncake is required for the Mooncake hidden-states backend. "
                "Install it with `pip install mooncake-transfer-engine`."
            ) from e

        store = MooncakeDistributedStore()
        _ensure_remove_force(store)
        rc = store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        if rc != 0:
            # Raising (instead of caching a half-dead client) lets callers with
            # lazy setup retry later, e.g. a producer that outlives a master
            # restart.
            raise RuntimeError(
                f"mooncake store setup failed (rc={rc}): master "
                f"{self.config.master_server_address} unreachable?"
            )
        self._store = store
        return self

    def _ensure_put_pool(self) -> _PinnedBufferPool:
        # Allocated lazily on first write so consumer-only instances (the
        # trainer) never pin producer buffers.
        if self._put_pool is None:
            with self._put_lock:
                if self._put_pool is None:
                    self._put_pool = _PinnedBufferPool(
                        self._store,
                        self.config.transfer_buffer_size,
                        self.config.transfer_pool_size,
                    )
        return self._put_pool

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        assert self._store is not None, "call setup() first"
        pool = self._ensure_put_pool()
        buf = pool.acquire()
        try:
            cuda_device = next(
                (t.device.index for t in tensors.values() if t.is_cuda), None
            )
            if cuda_device is not None:
                # Stage GPU tensors on the dedicated stream: ordered after the
                # kernels that produced them (ready event), off the compute
                # stream so vLLM's forward pass is never blocked, and complete
                # before the CPU-side put_from reads the buffer (done event).
                copy_stream = _get_staging_stream()
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(cuda_device))
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(ready)
                    size = _pack_into(buf, tensors, copy_stream=copy_stream)
                    done = torch.cuda.Event()
                    done.record(copy_stream)
                done.synchronize()
            else:
                size = _pack_into(buf, tensors)
            # Serialize store writes across writer threads (mooncake client
            # puts are not thread-safe; TorchSpec locks these the same way).
            with self._put_lock:
                rc = self._store.put_from(key, buf.data_ptr(), size)
            if rc != 0:
                raise RuntimeError(
                    f"mooncake put_from failed for {key} (rc={rc}). Note all "
                    "clients of one master must use the same protocol "
                    "(tcp vs rdma); mixed-protocol segments cannot transfer."
                )
        finally:
            pool.release(buf)

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        assert self._store is not None, "call setup() first"
        self._wait_exists(key, timeout, poll_interval)
        handle = self._store.get_buffer(key)
        if handle is None:
            raise RuntimeError(
                f"mooncake get_buffer returned no data for {key}: evicted "
                "(store full — is the delete path running?) or the producer's "
                "segment uses an incompatible protocol."
            )
        return _unpack_from(handle.ptr(), handle.size())

    def remove_sample(self, key: str) -> None:
        """Delete a consumed sample; otherwise the store fills up and puts stall."""
        assert self._store is not None, "call setup() first"
        self._store.remove(key, force=True)

    def _wait_exists(self, key: str, timeout: float, poll_interval: float) -> None:
        # Existence polls are metadata-only; the key normally already exists by
        # the time the trainer holds it, so this rarely loops.
        assert self._store is not None, "call setup() first"
        deadline = time.monotonic() + timeout
        while self._store.is_exist(key) != 1:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(poll_interval)
