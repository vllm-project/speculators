"""Balance and partition guarantees for MultipackDistributedBatchSamplerV2."""

import numpy as np
import pytest

from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)

MAX_LEN = 8192


def _skewed_lengths(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(mean=7.6, sigma=0.9, size=n).astype(int), 64, MAX_LEN)


def _shards(lengths: np.ndarray, replicas: int) -> list[list[np.ndarray]]:
    return [
        list(
            iter(
                MultipackDistributedBatchSamplerV2(
                    batch_max_length=MAX_LEN,
                    lengths=lengths,
                    num_replicas=replicas,
                    rank=rank,
                )
            )
        )
        for rank in range(replicas)
    ]


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_sample_counts_are_balanced_across_ranks(replicas):
    lengths = _skewed_lengths()
    counts = [sum(len(b) for b in batches) for batches in _shards(lengths, replicas)]

    assert min(counts) > 0
    assert max(counts) / min(counts) < 1.15, f"sample counts unbalanced: {counts}"


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_token_counts_stay_balanced_across_ranks(replicas):
    lengths = _skewed_lengths()
    tokens = [
        sum(int(lengths[b].sum()) for b in batches)
        for batches in _shards(lengths, replicas)
    ]

    assert max(tokens) / min(tokens) < 1.15, f"token counts unbalanced: {tokens}"


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_ranks_form_a_disjoint_partition(replicas):
    lengths = _skewed_lengths()
    shards = _shards(lengths, replicas)

    seen: list[set[int]] = []
    for batches in shards:
        idxs = [int(i) for b in batches for i in b]
        assert len(idxs) == len(set(idxs)), "duplicate index within a single rank"
        seen.append(set(idxs))

    for i in range(replicas):
        for j in range(i + 1, replicas):
            assert not (seen[i] & seen[j]), f"ranks {i}/{j} share samples"


@pytest.mark.parametrize("replicas", [2, 3, 4, 8])
def test_equal_batch_count_per_rank(replicas):
    counts = {len(batches) for batches in _shards(_skewed_lengths(), replicas)}
    assert len(counts) == 1, f"ranks disagree on batch count: {counts}"


@pytest.mark.parametrize("replicas", [2, 3])
def test_batches_respect_token_budget(replicas):
    lengths = _skewed_lengths()
    for batches in _shards(lengths, replicas):
        for b in batches:
            assert int(lengths[b].sum()) <= MAX_LEN


def test_rotation_actually_moves_the_largest_sample_off_rank_zero():
    lengths = _skewed_lengths()
    replicas = 3
    shards = _shards(lengths, replicas)
    nbatches = len(shards[0])

    owners = set()
    for i in range(min(nbatches, 30)):
        best_rank, best_len = None, -1
        for rank in range(replicas):
            if len(shards[rank][i]) == 0:
                continue
            m = int(lengths[shards[rank][i]].max())
            if m > best_len:
                best_rank, best_len = rank, m
        owners.add(best_rank)

    assert len(owners) > 1, "largest sample always lands on the same rank"
