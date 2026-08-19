"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1


class MooncakeIntegrityError(RuntimeError):
    """A Mooncake object was present but failed an integrity check."""


class NonFiniteTensorError(MooncakeIntegrityError):
    """A producer attempted to publish a tensor containing NaN or infinity."""


def _cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _tensor_checksum(tensor: torch.Tensor) -> str:
    """Return a fast checksum without copying the contiguous CPU tensor."""
    byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
    checksum = zlib.crc32(memoryview(byte_view)) & 0xFFFFFFFF
    return f"{checksum:08x}"


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
    """Mooncake's Python API returns negative status codes for some failures."""
    if isinstance(result, int) and result != 0:
        raise RuntimeError(
            f"Mooncake {operation} failed for key={key} with status={result}"
        )


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

    Each sample is written via ``put_tensor`` under ``{key}:{name}`` plus a
    versioned ``{key}:meta`` JSON manifest. The manifest includes shape, dtype,
    and CRC32 for every tensor and is written last, so its presence marks the
    sample complete and ``get_sample`` can poll for it.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None

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

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        if self._store is None:
            raise RuntimeError("call setup() first")

        prepared = {name: _cpu_contiguous(tensor) for name, tensor in tensors.items()}
        manifest_tensors: dict[str, dict[str, Any]] = {}
        for name, tensor in prepared.items():
            manifest_tensors[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "checksum": _tensor_checksum(tensor),
            }

        written_keys: list[str] = []
        meta_key = f"{key}:meta"
        try:
            for name, tensor in prepared.items():
                tensor_key = f"{key}:{name}"
                result = self._store.put_tensor(tensor_key, tensor)
                written_keys.append(tensor_key)
                _check_store_result("put_tensor", tensor_key, result)

            manifest = {
                "version": _MANIFEST_VERSION,
                "status": "ok",
                "tensors": manifest_tensors,
            }
            result = self._store.put(meta_key, json.dumps(manifest).encode("utf-8"))
            _check_store_result("put", meta_key, result)
        except Exception:
            # Do not leave partially published tensor objects behind. In particular,
            # never publish the completion manifest after a negative put status.
            cleanup_keys = [*written_keys, meta_key]
            if cleanup_keys:
                try:
                    self._store.batch_remove(cleanup_keys, force=True)
                except Exception:  # pragma: no cover - best-effort cleanup
                    logger.exception("Failed to clean partial Mooncake sample %s", key)
            raise

    def put_error(self, key: str, error: str) -> None:
        """Publish a small terminal marker so consumers fail fast and can retry."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        manifest = {
            "version": _MANIFEST_VERSION,
            "status": "error",
            "error": error[:4096],
            "tensors": {},
        }
        meta_key = f"{key}:meta"
        result = self._store.put(meta_key, json.dumps(manifest).encode("utf-8"))
        _check_store_result("put", meta_key, result)

    def delete_sample(self, key: str) -> None:
        """Remove all keys for a sample from the store."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        raw = self._store.get(f"{key}:meta")
        if not raw:
            return
        try:
            manifest = json.loads(raw)
            names = list(manifest.get("tensors", {}))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(
                "Corrupt manifest for key=%s; falling back to default tensor names", key
            )
            names = ["hidden_states", "token_ids"]
        keys_to_remove = [f"{key}:{name}" for name in names] + [f"{key}:meta"]
        results = self._store.batch_remove(keys_to_remove, force=True)
        for key_to_remove, status in zip(keys_to_remove, results, strict=True):
            _check_store_result("batch_remove", key_to_remove, status)

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        if self._store is None:
            raise RuntimeError("call setup() first")
        raw_manifest = self._wait_for(f"{key}:meta", timeout, poll_interval)
        tensor_specs = self._parse_manifest(key, raw_manifest)

        result = {}
        for name, spec in tensor_specs.items():
            tensor = self._store.get_tensor(f"{key}:{name}")
            if tensor is None:
                raise MooncakeIntegrityError(
                    f"Mooncake tensor unavailable for key={key}:{name}"
                )
            self._validate_tensor(key, name, tensor, spec)
            result[name] = tensor
        return result

    @staticmethod
    def _parse_manifest(key: str, raw_manifest: bytes) -> dict[str, dict[str, Any]]:
        try:
            manifest = json.loads(raw_manifest)
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
    def _validate_tensor(
        key: str,
        name: str,
        tensor: torch.Tensor,
        spec: dict[str, Any],
    ) -> None:
        if not isinstance(spec, dict):
            raise MooncakeIntegrityError(
                f"Invalid tensor manifest for key={key}:{name}: "
                f"expected object, got {type(spec).__name__}"
            )
        expected_shape = tuple(spec.get("shape", ()))
        expected_dtype = spec.get("dtype")
        if tuple(tensor.shape) != expected_shape:
            raise MooncakeIntegrityError(
                f"Mooncake shape mismatch for key={key}:{name}: "
                f"expected={expected_shape}, actual={tuple(tensor.shape)}"
            )
        if str(tensor.dtype) != expected_dtype:
            raise MooncakeIntegrityError(
                f"Mooncake dtype mismatch for key={key}:{name}: "
                f"expected={expected_dtype}, actual={tensor.dtype}"
            )
        expected_checksum = spec.get("checksum")
        actual_checksum = _tensor_checksum(_cpu_contiguous(tensor))
        if actual_checksum != expected_checksum:
            raise MooncakeIntegrityError(
                f"Mooncake checksum mismatch for key={key}:{name}: "
                f"expected={expected_checksum}, actual={actual_checksum}"
            )

    def _wait_for(self, key: str, timeout: float, poll_interval: float) -> bytes:
        if self._store is None:
            raise RuntimeError("call setup() first")
        deadline = time.monotonic() + timeout
        while True:
            raw = self._store.get(key)
            if raw:
                return raw
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(poll_interval)
