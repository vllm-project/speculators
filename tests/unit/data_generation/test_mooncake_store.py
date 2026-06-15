"""Unit tests for the Mooncake hidden-states store round-trip.

These exercise the producer/consumer payload contract without a real Mooncake
cluster by swapping in a dict-backed fake for ``MooncakeDistributedStore``.
The point is to prove the seam: a tensor dict written by the producer is read
back byte-identical by the consumer.
"""

import pytest
import torch

from speculators.data_generation.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
)


class _FakeMooncakeStore:
    """In-memory stand-in for MooncakeDistributedStore's bytes API."""

    def __init__(self):
        self.kv: dict[str, bytes] = {}

    def put(self, key: str, value: bytes) -> int:
        self.kv[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        return self.kv.get(key, b"")


@pytest.fixture
def store() -> MooncakeHiddenStatesStore:
    s = MooncakeHiddenStatesStore(MooncakeStoreConfig())
    s._store = _FakeMooncakeStore()  # bypass setup(); no real cluster needed
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


def test_meta_written_last_gates_visibility(store):
    # get_sample keys off the meta blob, which put_sample writes last. Simulate
    # a half-written sample (tensors present, meta absent) -> consumer waits.
    store._store.kv["req-2:hidden_states"] = b"partial"
    with pytest.raises(TimeoutError):
        store.get_sample("req-2", timeout=0.2, poll_interval=0.02)
