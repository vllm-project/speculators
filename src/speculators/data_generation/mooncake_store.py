"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import torch


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8080/metadata"
    master_server_address: str = "localhost:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 512 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is written via ``put_tensor`` under ``{key}:{name}`` plus a
    ``{key}:meta`` JSON marker listing tensor names. ``meta`` is written last,
    so its presence marks the sample complete and ``get_sample`` can poll for it.
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
            from mooncake.store import MooncakeDistributedStore  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Mooncake is required for the Mooncake hidden-states backend. "
                "Install it with `pip install mooncake-transfer-engine`."
            ) from e

        store = MooncakeDistributedStore()
        store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        self._store = store
        return self

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        assert self._store is not None, "call setup() first"
        names = []
        for name, tensor in tensors.items():
            self._store.put_tensor(
                f"{key}:{name}", tensor.detach().to("cpu").contiguous()
            )
            names.append(name)
        self._store.put(f"{key}:meta", json.dumps(names).encode("utf-8"))

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        assert self._store is not None, "call setup() first"
        names = json.loads(self._wait_for(f"{key}:meta", timeout, poll_interval))
        return {name: self._store.get_tensor(f"{key}:{name}") for name in names}

    def _wait_for(self, key: str, timeout: float, poll_interval: float) -> bytes:
        deadline = time.monotonic() + timeout
        while True:
            raw = self._store.get(key)
            if raw:
                return raw
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(poll_interval)
