"""Tests for the vLLM cache extraction used by the Mooncake connector."""

import pytest
import torch

pytest.importorskip("vllm")

from hs_connectors.mooncake_hidden_states_connector import (
    extract_from_kv_cache,
)


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_extract_from_kv_cache_preserves_hidden_state_slots(legacy_layout):
    """Extract token slots from both supported cache layouts."""
    num_blocks, num_hidden_states, block_size, hidden_size = (2, 4, 3, 2)
    logical_cache = torch.arange(
        num_blocks * num_hidden_states * block_size * hidden_size
    ).reshape(num_blocks, num_hidden_states, block_size, hidden_size)
    kv_cache = logical_cache.transpose(1, 2) if legacy_layout else logical_cache
    slot_mapping = torch.tensor([0, 1, 2, 3, 5])

    actual = extract_from_kv_cache(
        kv_cache,
        slot_mapping,
        num_tokens=4,
        block_size=block_size,
        num_hidden_states=num_hidden_states,
    )
    expected = torch.stack(
        [
            logical_cache[0, :, 0],
            logical_cache[0, :, 1],
            logical_cache[0, :, 2],
            logical_cache[1, :, 0],
        ]
    )

    assert actual.shape == (4, num_hidden_states, hidden_size)
    assert torch.equal(actual, expected)
