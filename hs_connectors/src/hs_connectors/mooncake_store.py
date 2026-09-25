"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from hs_connectors.device import ensure_accelerator_context

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 2
_TRAILER_LEN_BYTES = 8
_TENSOR_ALIGN = 64
_STAGING_GRANULE = 64 << 20
_OBJECT_NOT_FOUND = -704
_MIN_POLL_INTERVAL = 0.001


class MooncakeIntegrityError(RuntimeError):
    """A Mooncake object was present but failed an integrity check."""


class NonFiniteTensorError(MooncakeIntegrityError):
    """A producer attempted to publish a tensor containing NaN or infinity."""


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise ``NonFiniteTensorError`` if ``tensor`` holds NaN or infinity.

    ``amin``/``amax`` propagate NaN and surface either infinity, so the whole
    tensor is screened in two reductions with no full-size boolean temporary.
    Call this while the data is still on the accelerator: on the host it is an
    extra pass over the whole sample on the critical path of a producer write.
    """
    if not tensor.is_floating_point() or tensor.numel() == 0:
        return

    bounds = torch.stack((tensor.amin(), tensor.amax()))
    if bool(torch.isfinite(bounds).all()):
        return

    # Only pay for the exact counts once we already know the sample is bad.
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    raise NonFiniteTensorError(
        f"Non-finite producer tensor {name!r}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, nan_count={nan_count}, inf_count={inf_count}"
    )


def _check_store_result(operation: str, key: str, result: Any) -> None:
    """Mooncake's Python API returns negative status codes for failures."""
    if not isinstance(result, int) or result < 0:
        raise RuntimeError(
            f"Mooncake {operation} failed for key={key} with status={result}"
        )


def _align(offset: int) -> int:
    return -(-offset // _TENSOR_ALIGN) * _TENSOR_ALIGN


def _encode_trailer(manifest: dict[str, Any]) -> bytes:
    body = json.dumps(manifest).encode("utf-8")
    return body + len(body).to_bytes(_TRAILER_LEN_BYTES, "little")


def _layout(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, Any], int]:
    """Assign aligned offsets; return the manifest and where its trailer starts."""
    specs: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        specs[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "offset": offset,
            "nbytes": nbytes,
        }
        offset = _align(offset + nbytes)
    return {"version": _MANIFEST_VERSION, "status": "ok", "tensors": specs}, offset


def _parse_dtype(key: str, name: str, value: Any) -> torch.dtype:
    dtype = getattr(torch, str(value).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise MooncakeIntegrityError(f"Unknown dtype for key={key}:{name}: {value!r}")
    return dtype


def _empty_host(nbytes: int, pin: bool) -> torch.Tensor:
    if pin:
        try:
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except Exception:  # noqa: BLE001 - accelerator without pinned memory
            logger.warning("Pinned host memory unavailable; using pageable memory")
    return torch.empty(nbytes, dtype=torch.uint8)


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    master_server_address: str = "127.0.0.1:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 2 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 4

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            logger.warning("Unknown MooncakeStoreConfig keys ignored: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is one object: the 64-byte-aligned tensor bytes followed by a
    JSON manifest (shape, dtype, offset of every tensor) and its 8-byte length. A
    Mooncake put becomes visible atomically, so the object's presence marks the
    sample complete. Objects move through per-thread staging buffers that are
    registered with Mooncake once, so ``put_from``/``get_into`` go zero-copy
    over RDMA and skip the client-side bounce buffer.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None
        self._staging: dict[tuple[int, str], torch.Tensor] = {}

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
                "Install it with `pip install mooncake-transfer-engine` or "
                "`pip install mooncake-transfer-engine-cuda13`."
            ) from e

        store = MooncakeDistributedStore()
        ensure_accelerator_context()
        result = store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        _check_store_result("setup", self.config.local_hostname, result)
        self._store = store
        return self

    def _require_store(self) -> Any:
        if self._store is None:
            raise RuntimeError("call setup() first")
        return self._store

    def _staging_buffer(self, role: str, nbytes: int) -> torch.Tensor:
        """This thread's registered buffer for ``role``, grown to ``nbytes``."""
        store = self._require_store()
        slot = (threading.get_ident(), role)
        buffer = self._staging.get(slot)
        if buffer is not None and buffer.numel() >= nbytes:
            return buffer
        if buffer is not None:
            result = store.unregister_buffer(buffer.data_ptr())
            if result != 0:
                logger.warning("Mooncake unregister_buffer returned %s", result)
        capacity = -(-max(nbytes, 1) // _STAGING_GRANULE) * _STAGING_GRANULE
        buffer = _empty_host(capacity, pin=role == "put")
        _check_store_result(
            "register_buffer",
            f"<{role} staging>",
            store.register_buffer(buffer.data_ptr(), capacity),
        )
        self._staging[slot] = buffer
        return buffer

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        """Publish ``tensors``, which may live on the accelerator.

        Device tensors are copied into the pinned staging buffer on the current
        stream, so callers can pick the stream the DtoH copy runs on.
        """
        store = self._require_store()
        manifest, trailer_offset = _layout(tensors)
        trailer = _encode_trailer(manifest)
        total = trailer_offset + len(trailer)
        staging = self._staging_buffer("put", total)
        for name, tensor in tensors.items():
            spec = manifest["tensors"][name]
            region = staging[spec["offset"] : spec["offset"] + spec["nbytes"]]
            region.view(tensor.dtype).view(tensor.shape).copy_(tensor)
        staging[trailer_offset:total].copy_(
            torch.frombuffer(bytearray(trailer), dtype=torch.uint8)
        )

        result = store.put_from(key, staging.data_ptr(), total)
        _check_store_result("put_from", key, result)

    def put_error(self, key: str, error: str) -> None:
        """Publish a small terminal marker so consumers fail fast and can retry."""
        store = self._require_store()
        manifest = {
            "version": _MANIFEST_VERSION,
            "status": "error",
            "error": error[:4096],
            "tensors": {},
        }
        _check_store_result("put", key, store.put(key, _encode_trailer(manifest)))

    def delete_sample(self, key: str) -> None:
        """Remove a sample from the store; a missing sample is not an error."""
        result = self._require_store().remove(key, force=True)
        if result != _OBJECT_NOT_FOUND:
            _check_store_result("remove", key, result)

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        store = self._require_store()
        self._wait_for(key, timeout, poll_interval)

        size = store.get_size(key)
        if not isinstance(size, int) or size < 0:
            raise MooncakeIntegrityError(
                f"Mooncake sample unavailable for key={key} (status={size})"
            )
        staging = self._staging_buffer("get", size)
        received = store.get_into(key, staging.data_ptr(), size)
        if received != size:
            raise MooncakeIntegrityError(
                f"Mooncake get_into for key={key} returned {received}, "
                f"expected {size} bytes"
            )

        tensor_specs = self._parse_manifest(key, staging[:size])
        return {
            name: self._read_tensor(key, name, staging[:size], spec)
            for name, spec in tensor_specs.items()
        }

    @staticmethod
    def _parse_manifest(key: str, obj: torch.Tensor) -> dict[str, dict[str, Any]]:
        size = obj.numel()
        trailer_len = (
            int.from_bytes(obj[-_TRAILER_LEN_BYTES:].numpy().tobytes(), "little")
            if size >= _TRAILER_LEN_BYTES
            else size
        )
        if trailer_len + _TRAILER_LEN_BYTES > size:
            raise MooncakeIntegrityError(
                f"Truncated Mooncake manifest for key={key}: "
                f"object={size} bytes, manifest={trailer_len} bytes"
            )
        manifest_end = size - _TRAILER_LEN_BYTES
        raw = obj[manifest_end - trailer_len : manifest_end].numpy().tobytes()
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MooncakeIntegrityError(
                f"Corrupt Mooncake manifest for key={key}: {e}"
            ) from e

        if not isinstance(manifest, dict):
            raise MooncakeIntegrityError(
                f"Invalid Mooncake manifest type for key={key}: "
                f"{type(manifest).__name__}"
            )
        if manifest.get("status") == "error":
            raise MooncakeIntegrityError(
                f"Mooncake producer rejected key={key}: "
                f"{manifest.get('error', 'unknown producer error')}"
            )
        if manifest.get("version") != _MANIFEST_VERSION:
            raise MooncakeIntegrityError(
                f"Unsupported Mooncake manifest version for key={key}: "
                f"{manifest.get('version')!r}"
            )
        tensor_specs = manifest.get("tensors")
        if not isinstance(tensor_specs, dict) or not tensor_specs:
            raise MooncakeIntegrityError(
                f"Mooncake manifest has no tensors for key={key}"
            )
        return tensor_specs

    @staticmethod
    def _read_tensor(key: str, name: str, obj: torch.Tensor, spec: Any) -> torch.Tensor:
        """Copy one tensor out of the staging buffer, which the next get reuses."""
        try:
            shape = tuple(int(d) for d in spec["shape"])
            dtype = _parse_dtype(key, name, spec["dtype"])
            offset, nbytes = int(spec["offset"]), int(spec["nbytes"])
        except (KeyError, TypeError, ValueError) as e:
            raise MooncakeIntegrityError(
                f"Invalid tensor manifest for key={key}:{name}: {spec!r}"
            ) from e

        expected = torch.Size(shape).numel() * dtype.itemsize
        if nbytes != expected or offset < 0 or offset + nbytes > obj.numel():
            raise MooncakeIntegrityError(
                f"Mooncake tensor out of bounds for key={key}:{name}: "
                f"offset={offset}, nbytes={nbytes}, expected_nbytes={expected}, "
                f"object={obj.numel()} bytes"
            )
        return obj[offset : offset + nbytes].view(dtype).view(shape).clone()

    def _wait_for(self, key: str, timeout: float, poll_interval: float) -> None:
        """Poll with exponential backoff capped at ``poll_interval``.

        vLLM answers the HTTP request before the connector's put completes, so
        the sample usually lands a few milliseconds after the consumer asks.
        """
        store = self._require_store()
        deadline = time.monotonic() + timeout
        delay = min(_MIN_POLL_INTERVAL, poll_interval)
        while True:
            exists = store.is_exist(key)
            if exists == 1:
                return
            if exists != 0:
                raise RuntimeError(
                    f"Mooncake is_exist failed for key={key} with status={exists}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(delay)
            delay = min(delay * 2, poll_interval)
