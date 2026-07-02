"""Unit tests for the Mooncake hidden-states store round-trip.

These exercise the producer/consumer payload contract without a real Mooncake
cluster by swapping in a dict-backed fake for ``MooncakeDistributedStore``.
The fake models the zero-copy buffer API (``put_from``/``get_buffer``), so the
real pack/unpack code path is covered: a tensor dict staged into a buffer by the
producer is read back byte-identical by the consumer.
"""

import ctypes

import pytest
import torch

# hs_connectors is an optional dependency (the mooncake extra); skip when absent.
pytest.importorskip("hs_connectors.mooncake_store")

from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
    _ensure_remove_force,
    resolve_local_hostname,
)


class _FakeHandle:
    """Stand-in for mooncake.store.BufferHandle."""

    def __init__(self, ptr: int, size: int):
        self._ptr = ptr
        self._size = size

    def ptr(self) -> int:
        return self._ptr

    def size(self) -> int:
        return self._size


class _FakeMooncakeStore:
    """In-memory stand-in modeling the zero-copy buffer API."""

    def __init__(self):
        self._blobs: dict[str, bytes] = {}
        self._keepalive: list = []  # keep get_buffer backing memory alive

    def register_buffer(self, ptr: int, size: int) -> int:
        return 0

    def put_from(self, key: str, ptr: int, size: int) -> int:
        self._blobs[key] = ctypes.string_at(ptr, size)
        return 0

    def is_exist(self, key: str) -> int:
        return 1 if key in self._blobs else 0

    def get_buffer(self, key: str):
        data = self._blobs[key]
        cbuf = ctypes.create_string_buffer(data, len(data))
        self._keepalive.append(cbuf)
        return _FakeHandle(ctypes.addressof(cbuf), len(data))

    def remove(self, key: str, force: bool = False) -> int:
        self._blobs.pop(key, None)
        return 0


@pytest.fixture
def store() -> MooncakeHiddenStatesStore:
    # Tiny transfer buffer/pool: test samples are a few KB.
    s = MooncakeHiddenStatesStore(
        MooncakeStoreConfig(transfer_buffer_size=1 << 20, transfer_pool_size=1)
    )
    # bypass setup(); no real cluster needed
    s._store = _FakeMooncakeStore()  # type: ignore[assignment]
    return s


def test_put_get_roundtrip_preserves_shape_and_dtype(store):
    # Mirrors the ExampleHiddenStatesConnector payload: [seq, n_layers, hidden]
    # bf16 hidden states + int64 token ids.
    hidden_states = torch.randn(7, 4, 16, dtype=torch.bfloat16)
    token_ids = torch.arange(7, dtype=torch.int64)

    store.put_sample("req-1", {"hidden_states": hidden_states, "token_ids": token_ids})
    out = store.get_sample("req-1", timeout=1.0)

    assert out.keys() == {"hidden_states", "token_ids"}
    assert out["hidden_states"].shape == hidden_states.shape
    assert out["hidden_states"].dtype == torch.bfloat16
    assert torch.equal(out["hidden_states"], hidden_states)
    assert torch.equal(out["token_ids"], token_ids)


def test_missing_key_times_out(store):
    # get_sample polls is_exist; a key that was never written must time out.
    with pytest.raises(TimeoutError):
        store.get_sample("never-written", timeout=0.2, poll_interval=0.02)


def test_remove_sample_frees_the_key(store):
    # on_generate="delete" relies on remove_sample; without it consumed samples
    # accumulate until the store hits capacity and producer puts stall.
    store.put_sample("req-3", {"token_ids": torch.arange(4)})
    store.remove_sample("req-3")

    with pytest.raises(TimeoutError):
        store.get_sample("req-3", timeout=0.2, poll_interval=0.02)


def test_resolve_local_hostname_is_routable():
    # Auto-detection must never return loopback for a remote master: the
    # advertised address is what remote peers connect back to.
    addr = resolve_local_hostname("8.8.8.8:50051")
    assert not addr.startswith("127.")

    # A loopback master is a single-node deployment: loopback is correct.
    assert resolve_local_hostname("127.0.0.1:50051").startswith("127.")


def test_resolve_local_hostname_env_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.0.0.99")
    assert resolve_local_hostname("8.8.8.8:50051") == "10.0.0.99"


def test_for_consumer_mounts_no_segment():
    # Read-only clients must not contribute store capacity; placement then
    # only ever targets producer segments.
    cfg = MooncakeStoreConfig.for_consumer(protocol="rdma")
    assert cfg.global_segment_size == 0
    assert cfg.protocol == "rdma"
    assert cfg.local_buffer_size > 0


def test_ensure_remove_force_accepts_force_kwarg():
    class _Store:
        def remove(self, key, force=False):
            return 0

    _ensure_remove_force(_Store())  # must not raise


def test_ensure_remove_force_rejects_old_builds():
    class _OldStore:
        def remove(self, key):
            return 0

    with pytest.raises(RuntimeError, match="0.3.10.post1"):
        _ensure_remove_force(_OldStore())
