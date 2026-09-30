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
from math import prod
from typing import Any

import torch

from hs_connectors.device import ensure_accelerator_context

logger = logging.getLogger(__name__)


VERSION = 1

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



def packed_layout(
    tensors: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> tuple[dict[str, Any], int]:
    """Assign aligned offsets; return the tensor spec and total size"""
    specs: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, tensor_metadata in tensors.items():
        shape, dtype = tensor_metadata
        nbytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
        specs[name] = {
            "shape": shape,
            "dtype": str(dtype),
            "nbytes": nbytes,
            "offset": offset,  # byte offset in packed tensor
        }
        offset = _align(offset + nbytes)
    return specs, offset


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
                "Install it with `pip install 'mooncake-transfer-engine>=0.3.12'` or "
                "`pip install 'mooncake-transfer-engine-cuda13>=0.3.12'`."
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
            del self._staging[slot]
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

    def put_sample(
        self, transfer_manifest: dict[str, Any], tensors: dict[str, torch.Tensor]
    ) -> None:
        """Publish ``tensors``, which may live on the accelerator.

        Device tensors are copied into the pinned staging buffer on the current
        stream, so callers can pick the stream the DtoH copy runs on.
        """
        store = self._require_store()
        tensor_spec = transfer_manifest["tensors"]
        total_bytes = transfer_manifest["metadata"]["total_aligned_bytes"]
        staging = self._staging_buffer("put", total_bytes)

        for name, spec in tensor_spec.items():
            tensor = tensors[name]
            region = staging[spec["offset"] : spec["offset"] + spec["nbytes"]]
            region.view(tensor.dtype).view(tensor.shape).copy_(tensor)

        handle = transfer_manifest["handle"]
        result = store.put_from(handle, staging.data_ptr(), total_bytes)
        _check_store_result("put_from", handle, result)

    def put_error(self, transfer_manifest: dict[str, Any], error: str) -> None:
        """Publish a small terminal marker so consumers fail fast and can retry."""
        store = self._require_store()
        error_dict = {
            "error": error[:4096],
        }
        error_handle = transfer_manifest["handle"] + ":error"

        _check_store_result(
            "put", error_handle, store.put(error_handle, json.dumps(error_dict).encode("utf-8"))
        )

    def delete_sample(self, key: str) -> None:
        """Remove a sample from the store; a missing sample is not an error."""
        error_handle = key + ":error"

        for k in (key, error_handle):
            result = self._require_store().remove(k, force=True)
            if result != _OBJECT_NOT_FOUND:
                _check_store_result("remove", k, result)


    def get_sample(
        self, transfer_manifest: dict[str, Any], timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
#         Pseudocode callgraph:
#
#         get_sample()
#             _wait_for()
#                 blocks until is_exist(handle) or is_exist(error_handle)
#                 if is_exist(error_handle):
#                     _handle_error()
#                         _read_from_store(error_handle)
#                         raise Error
#
#             staging = _read_from_store(handle)
# for tensor in transfer_manifest["tensors"]:
#                 _extract_tensor(tensor, staging)
#
#             return extracted tensors

        if (manifest_version := transfer_manifest.get("version", 0)) != VERSION:
            raise MooncakeIntegrityError(
                f"Mooncake manifest version {manifest_version} didn't match local version {VERSION}"
                f"Transfer manifest: {transfer_manifest}\n"
            )

        self._wait_for(transfer_manifest, timeout, poll_interval)

        handle = transfer_manifest["handle"]
        tensor_specs = transfer_manifest["tensors"]
        staging_tensor = self._read_from_store(handle, transfer_manifest["metadata"]["total_aligned_bytes"])

        return {
            name: self._extract_tensor(handle, name, staging_tensor, spec)
            for name, spec in tensor_specs.items()
        }

    def _read_from_store(self, key, expected_payload_size: int = 0):
        store = self._require_store()
        size = store.get_size(key)
        if not isinstance(size, int) or size < 0:
            raise MooncakeIntegrityError(
                f"Mooncake value unavailable for key={key} (status={size})"
            )
        if expected_payload_size > 0 and size != expected_payload_size:
            raise MooncakeIntegrityError(
                f"Mooncake payload size {size} doesn't match expected payload size {expected_payload_size}"
            )

        staging = self._staging_buffer("get", size)
        received = store.get_into(key, staging.data_ptr(), size)
        if received != size:
            raise MooncakeIntegrityError(
                f"Mooncake get_into for key={key} returned {received}, "
                f"expected {size} bytes"
            )

        return staging[:size]

    def _process_error(self, error_handle, transfer_manifest: dict[str, Any]):

        obj = self._read_from_store(error_handle)

        raw = obj.numpy().tobytes()
        try:
            error_dict =  json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MooncakeIntegrityError(
                f"Corrupt Mooncake json for key={error_handle}: {e}"
            ) from e

        if not isinstance(error_dict, dict):
            raise MooncakeIntegrityError(
                f"Invalid Mooncake error manifest type for key={error_handle}: "
                f"Recieved: {error_dict}\n"
                f"Transfer manifest: {transfer_manifest}\n"
            )
        raise MooncakeIntegrityError(
            f"Mooncake producer rejected key={error_handle}: "
            f"{error_dict.get('error', 'unknown producer error')}\n"
            f"Transfer manifest: {transfer_manifest}\n"
        )

    @staticmethod
    def _extract_tensor(key: str, name: str, obj: torch.Tensor, spec: Any) -> torch.Tensor:
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

    def _wait_for(self, transfer_manifest: dict[str, Any], timeout: float, poll_interval: float) -> None:
        """Poll with exponential backoff capped at ``poll_interval``.

        vLLM answers the HTTP request before the connector's put completes, so
        the sample usually lands a few milliseconds after the consumer asks.

        Return True if success, return False if foun
        """
        store = self._require_store()

        def _check_key(key: str) -> bool:
            exists = store.is_exist(key)
            if exists == 1:
                return True
            if exists != 0:
                raise RuntimeError(
                    f"Mooncake is_exist failed for key={key} with status={exists}"
                )
            return False

        handle = transfer_manifest["handle"]
        error_handle = handle + ":error"

        deadline = time.monotonic() + timeout
        delay = min(_MIN_POLL_INTERVAL, poll_interval)
        while True:
            # Alternate checking for handle and error_handle
            if _check_key(error_handle):
                self._process_error(error_handle, transfer_manifest)
            if _check_key(handle):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {handle}")
            time.sleep(delay)
            delay = min(delay * 2, poll_interval)
