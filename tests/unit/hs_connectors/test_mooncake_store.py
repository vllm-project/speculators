"""Unit tests for the Mooncake hidden-states store round-trip.

These exercise the producer/consumer payload contract without a real Mooncake
cluster by swapping in a dict-backed fake for ``MooncakeDistributedStore``.
The point is to prove the seam: a tensor dict written by the producer is read
back byte-identical by the consumer.
"""

import ctypes

import pytest
import torch

# hs_connectors is an optional dependency (the mooncake extra); skip when absent.
pytest.importorskip("hs_connectors.mooncake_store")

from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeIntegrityError,
    MooncakeStoreConfig,
    NonFiniteTensorError,
    assert_finite,
)

from hs_connectors import mooncake_store


class _FakeMooncakeStore:
    """In-memory stand-in for MooncakeDistributedStore's pointer-based API."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.registered: dict[int, int] = {}

    def register_buffer(self, ptr: int, size: int) -> int:
        self.registered[ptr] = size
        return 0

    def unregister_buffer(self, ptr: int) -> int:
        return 0 if self.registered.pop(ptr, None) is not None else -1

    def _check_registered(self, ptr: int, size: int) -> None:
        assert any(
            base <= ptr and ptr + size <= base + length
            for base, length in self.registered.items()
        ), "transfer from unregistered memory"

    def put(self, key: str, value: bytes) -> int:
        if key in self.objects:
            return -705
        self.objects[key] = bytes(value)
        return 0

    def put_from(self, key: str, ptr: int, size: int) -> int:
        self._check_registered(ptr, size)
        return self.put(key, ctypes.string_at(ptr, size))

    def is_exist(self, key: str) -> int:
        return int(key in self.objects)

    def get_size(self, key: str) -> int:
        return len(self.objects[key]) if key in self.objects else -704

    def get_into(self, key: str, ptr: int, size: int) -> int:
        self._check_registered(ptr, size)
        data = self.objects.get(key)
        if data is None:
            return -704
        if len(data) > size:
            return -600
        ctypes.memmove(ptr, data, len(data))
        return len(data)

    def remove(self, key: str, force: bool = False) -> int:
        return 0 if self.objects.pop(key, None) is not None else -704


@pytest.fixture
def store() -> MooncakeHiddenStatesStore:
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
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


def test_missing_sample_times_out(store):
    with pytest.raises(TimeoutError):
        store.get_sample("req-2", timeout=0.2, poll_interval=0.02)


def test_delete_sample_removes_object(store):
    hs = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    tids = torch.arange(4, dtype=torch.int64)
    store.put_sample("req-del", {"hidden_states": hs, "token_ids": tids})

    store.delete_sample("req-del")

    assert store._store.objects == {}


def test_delete_sample_noop_when_missing(store):
    store.delete_sample("nonexistent-key")


def test_delete_sample_raises_on_negative_status(store, monkeypatch):
    monkeypatch.setattr(store._store, "remove", lambda _key, force: -800)

    with pytest.raises(RuntimeError, match="status=-800"):
        store.delete_sample("req-remove-fail")


def test_get_sample_raises_on_evicted_sample(store, monkeypatch):
    store.put_sample("req-evict", {"hidden_states": torch.zeros(4, 2, 8)})
    monkeypatch.setattr(store._store, "get_size", lambda _key: -704)

    with pytest.raises(MooncakeIntegrityError, match="unavailable"):
        store.get_sample("req-evict", timeout=1.0)


def test_get_sample_raises_on_short_read(store, monkeypatch):
    store.put_sample("req-short", {"hidden_states": torch.zeros(4, 2, 8)})
    monkeypatch.setattr(store._store, "get_into", lambda _key, _ptr, _size: 16)

    with pytest.raises(MooncakeIntegrityError, match="returned 16"):
        store.get_sample("req-short", timeout=1.0)


def test_get_sample_raises_on_is_exist_error(store, monkeypatch):
    monkeypatch.setattr(store._store, "is_exist", lambda _key: -1)

    with pytest.raises(RuntimeError, match="status=-1"):
        store.get_sample("req-exist-fail", timeout=1.0)


def test_get_sample_rejects_out_of_bounds_manifest(store):
    trailer = mooncake_store._encode_trailer(
        {
            "version": mooncake_store._MANIFEST_VERSION,
            "status": "ok",
            "tensors": {
                "hidden_states": {
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "offset": 0,
                    "nbytes": 4096,
                }
            },
        }
    )
    store._store.objects["req-oob"] = trailer

    with pytest.raises(MooncakeIntegrityError, match="out of bounds"):
        store.get_sample("req-oob", timeout=1.0)


def test_staging_buffer_is_reused_and_grows(store, monkeypatch):
    monkeypatch.setattr(mooncake_store, "_STAGING_GRANULE", 4096)
    store.put_sample("small-1", {"hidden_states": torch.zeros(16)})
    first = dict(store._store.registered)
    store.put_sample("small-2", {"hidden_states": torch.zeros(16)})
    assert store._store.registered == first

    store.put_sample("large", {"hidden_states": torch.zeros(4096)})
    assert len(store._store.registered) == 1
    assert next(iter(store._store.registered.values())) >= 4096 * 4
    assert torch.equal(
        store.get_sample("large", timeout=1.0)["hidden_states"], torch.zeros(4096)
    )


@pytest.mark.parametrize("bad_value", [torch.nan, torch.inf, -torch.inf])
def test_assert_finite_rejects_non_finite_tensors(bad_value):
    # The producer calls this on the accelerator before the DtoH copy, so
    # put_sample itself no longer walks the sample looking for NaN/inf.
    hs = torch.zeros(4, 2, 8, dtype=torch.bfloat16)
    hs[1, 0, 0] = bad_value

    with pytest.raises(NonFiniteTensorError, match="Non-finite producer tensor"):
        assert_finite("hidden_states", hs)


def test_assert_finite_accepts_clean_and_non_float_tensors():
    assert_finite("hidden_states", torch.randn(4, 2, 8, dtype=torch.bfloat16))
    assert_finite("token_ids", torch.arange(4, dtype=torch.int64))
    assert_finite("empty", torch.empty(0, dtype=torch.bfloat16))


def test_error_manifest_fails_consumer_immediately(store):
    store.put_error("req-error", "producer found NaN")

    with pytest.raises(MooncakeIntegrityError, match="producer found NaN"):
        store.get_sample("req-error", timeout=1.0)


def test_negative_put_status_does_not_publish_sample(store, monkeypatch):
    monkeypatch.setattr(store._store, "put_from", lambda _key, _ptr, _size: -800)

    with pytest.raises(RuntimeError, match="status=-800"):
        store.put_sample(
            "req-put-fail",
            {
                "hidden_states": torch.zeros(4, 2, 8),
                "token_ids": torch.arange(4),
            },
        )

    assert store._store.is_exist("req-put-fail") == 0


def test_register_failure_raises(store, monkeypatch):
    monkeypatch.setattr(store._store, "register_buffer", lambda _ptr, _size: -1)

    with pytest.raises(RuntimeError, match="register_buffer"):
        store.put_sample("req-reg-fail", {"hidden_states": torch.zeros(4)})


def test_wait_backs_off_from_one_millisecond(store, monkeypatch):
    sleeps: list[float] = []
    calls = iter([0, 0, 0, 0, 1])
    monkeypatch.setattr(store._store, "is_exist", lambda _key: next(calls))
    monkeypatch.setattr(mooncake_store.time, "sleep", sleeps.append)

    store._wait_for("req-late", timeout=10.0, poll_interval=0.004)

    assert sleeps == [0.001, 0.002, 0.004, 0.004]
